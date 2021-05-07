use cgmath::Point3;
use engine3d::{collision, events::*, geom::*, render::InstanceGroups, run, Engine, DT};
use rand;
use std::{f32::consts::PI, usize};
use winit;

const NUM_MARBLES: usize = 10;
const NUM_TERRAIN_BOXES: usize = 10;
const G: f32 = 9.8;

#[derive(Clone, Debug)]
pub struct Player {
    pub body: Sphere,
    pub velocity: Vec3,
    pub acc: Vec3,
    pub rot: Quat,
    pub omega: Vec3,
    pub hp:usize,
}

// TODO: implement player info
impl Player {
    const MAX_SPEED: f32 = 3.0;
    fn render(&self, rules: &GameData, igs: &mut InstanceGroups) {
        igs.render(
            rules.player_model,
            engine3d::render::InstanceRaw {
                model: (Mat4::from_translation(self.body.c.to_vec() - Vec3::new(0.0, 0.2, 0.0))
                    * Mat4::from_scale(self.body.r)
                    * Mat4::from(self.rot))
                .into(),
            },
        );
    }
    fn integrate(&mut self) {
        self.velocity += ((self.rot * self.acc) + Vec3::new(0.0, -G, 0.0)) * DT;
        if self.velocity.magnitude() > Self::MAX_SPEED {
            self.velocity = self.velocity.normalize_to(Self::MAX_SPEED);
        }
        self.body.c += self.velocity * DT;
        self.rot += 0.5 * DT * Quat::new(0.0, self.omega.x, self.omega.y, self.omega.z) * self.rot;
    }
}

// TODO: create a desirable camera
trait Camera {
    fn new() -> Self;
    fn update(&mut self, _events: &engine3d::events::Events, _player: &Player) {}
    fn render(&self, _rules: &GameData, _igs: &mut InstanceGroups) {}
    fn update_camera(&self, _cam: &mut engine3d::camera::Camera) {}
    fn integrate(&mut self) {}
}

#[derive(Clone, Debug)]
pub struct FPCamera {
    pub pitch: f32,
    player_pos: Pos3,
    player_rot: Quat,
}

impl Camera for FPCamera {
    fn new() -> Self {
        Self {
            pitch: 0.0,
            player_pos: Pos3::new(0.0, 0.0, 0.0),
            player_rot: Quat::new(1.0, 0.0, 0.0, 0.0),
        }
    }
    fn update(&mut self, events: &engine3d::events::Events, player: &Player) {
        let (_dx, dy) = events.mouse_delta();
        self.pitch += dy / 100.0;
        self.pitch = self.pitch.clamp(-PI / 4.0, PI / 4.0);
        self.player_pos = player.body.c;
        self.player_rot = player.rot;
    }
    fn update_camera(&self, c: &mut engine3d::camera::Camera) {
        c.eye = self.player_pos + Vec3::new(0.0, 0.5, 0.0);
        // The camera is pointing at a point just in front of the composition of the player's rot and the camera's rot (player * cam * forward-offset)
        let rotation = self.player_rot
            * (Quat::from(cgmath::Euler::new(
                cgmath::Rad(self.pitch),
                cgmath::Rad(0.0),
                cgmath::Rad(0.0),
            )));
        let offset = rotation * Vec3::unit_z();
        c.target = c.eye + offset;
    }
}

#[derive(Clone, Debug)]
pub struct OrbitCamera {
    pub pitch: f32,
    pub yaw: f32,
    pub distance: f32,
    player_pos: Pos3,
    player_rot: Quat,
}

impl Camera for OrbitCamera {
    fn new() -> Self {
        Self {
            pitch: 0.0,
            yaw: 0.0,
            distance: 5.0,
            player_pos: Pos3::new(0.0, 0.0, 0.0),
            player_rot: Quat::new(1.0, 0.0, 0.0, 0.0),
        }
    }
    fn update(&mut self, events: &engine3d::events::Events, player: &Player) {
        let (dx, dy) = events.mouse_delta();
        self.pitch += dy / 100.0;
        self.pitch = self.pitch.clamp(-PI / 4.0, PI / 4.0);

        self.yaw += dx / 100.0;
        self.yaw = self.yaw.clamp(-PI / 4.0, PI / 4.0);
        if events.key_pressed(KeyCode::Up) {
            self.distance -= 0.5;
        }
        if events.key_pressed(KeyCode::Down) {
            self.distance += 0.5;
        }
        self.player_pos = player.body.c;
        self.player_rot = player.rot;
        // TODO: when player moves, slightly move yaw towards zero
    }
    fn update_camera(&self, c: &mut engine3d::camera::Camera) {
        // The camera should point at the player
        c.target = self.player_pos;
        // And rotated around the player's position and offset backwards
        let camera_rot = self.player_rot
            * Quat::from(cgmath::Euler::new(
                cgmath::Rad(self.pitch),
                cgmath::Rad(self.yaw),
                cgmath::Rad(0.0),
            ));
        let offset = camera_rot * Vec3::new(0.0, 0.0, -self.distance);
        c.eye = self.player_pos + offset;
        // To be fancy, we'd want to make the camera's eye to be an object in the world and whose rotation is locked to point towards the player, and whose distance from the player is locked, and so on---so we'd have player OR camera movements apply accelerations to the camera which could be "beaten" by collision.
    }
}

#[derive(Clone, Debug)]
pub struct Marbles {
    pub body: Vec<Sphere>,
    pub velocity: Vec<Vec3>,
    pub hp: Vec<usize>
}

// Ziang: I think we can base our game with marbles & boxes...
impl Marbles {
    fn render(&self, rules: &GameData, igs: &mut InstanceGroups) {
        igs.render_batch(
            rules.marble_model,
            self.body.iter().map(|body| engine3d::render::InstanceRaw {
                model: (Mat4::from_translation(body.c.to_vec()) * Mat4::from_scale(body.r)).into(),
            }),
        );
    }
    fn integrate(&mut self) {
        for vel in self.velocity.iter_mut() {
            *vel += Vec3::new(0.0, -G, 0.0) * DT;
        }
        for (body, vel) in self.body.iter_mut().zip(self.velocity.iter()) {
            body.c += vel * DT;
        }
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = (&mut Sphere, &mut Vec3)> {
        self.body.iter_mut().zip(self.velocity.iter_mut())
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Wall {
    pub body: Plane,
    control: (i8, i8),
}

impl Wall {
    fn render(&self, rules: &GameData, igs: &mut InstanceGroups) {
        igs.render(
            rules.wall_model,
            engine3d::render::InstanceRaw {
                model: (Mat4::from(cgmath::Quaternion::between_vectors(
                    Vec3::new(0.0, 1.0, 0.0),
                    self.body.n,
                )) * Mat4::from_translation(Vec3::new(0.0, -0.025, 0.0))
                    * Mat4::from_nonuniform_scale(0.5, 0.05, 0.5))
                .into(),
            },
        );
    }

    fn input(&mut self, events: &engine3d::events::Events) {
        self.control.0 = if events.key_held(KeyCode::A) {
            -1
        } else if events.key_held(KeyCode::D) {
            1
        } else {
            0
        };
        self.control.1 = if events.key_held(KeyCode::W) {
            -1
        } else if events.key_held(KeyCode::S) {
            1
        } else {
            0
        };
    }
    fn integrate(&mut self) {
        self.body.n += Vec3::new(
            self.control.0 as f32 * 0.4 * DT,
            0.0,
            self.control.1 as f32 * 0.4 * DT,
        );
        self.body.n = self.body.n.normalize();
    }
}

#[derive(Clone, Debug)]
pub struct Terrain_Boxes {
    pub body: Vec<Box>,
    pub velocity: Vec<Vec3>,
    pub hp: Vec<usize>,
}

impl Terrain_Boxes {
    fn add_standard(&mut self, pos: Pos3) {
        let x_axis = Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let y_axis = Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        let z_axis = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        };
        let half_sizes = Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        };
        self.body.push(Box {
            c: pos,
            axes: Mat3 {
                x: x_axis,
                y: y_axis,
                z: z_axis,
            },
            half_sizes,
        });
        self.velocity.push(Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
    }

    fn render(&self, rules: &GameData, igs: &mut InstanceGroups) {
        igs.render_batch(
            rules.terrain_box_model,
            self.body.iter().map(|body| engine3d::render::InstanceRaw {
                model: (Mat4::from_translation(body.c.to_vec())
                    * Mat4::from_scale(body.half_sizes.x))
                .into(),
            }),
        );
    }
    fn integrate(&mut self) {
        for vel in self.velocity.iter_mut() {
            *vel += Vec3::new(0.0, -G, 0.0) * DT;
        }
        for (body, vel) in self.body.iter_mut().zip(self.velocity.iter()) {
            body.c += vel * DT;
        }
    }
    fn iter_mut(&mut self) -> impl Iterator<Item = (&mut Box, &mut Vec3)> {
        self.body.iter_mut().zip(self.velocity.iter_mut())
    }
}

// Ziang: should we allow for
struct Game<Cam: Camera> {
    marbles: Marbles,
    wall: Wall,
    terrain_boxes: Terrain_Boxes,
    player: Player,
    camera: Cam,
    pm: Vec<collision::Contact<usize>>,
    pw: Vec<collision::Contact<usize>>,
    mm: Vec<collision::Contact<usize>>,
    mw: Vec<collision::Contact<usize>>,
    tw: Vec<collision::Contact<usize>>,
}
struct GameData {
    marble_model: engine3d::assets::ModelRef,
    wall_model: engine3d::assets::ModelRef,
    player_model: engine3d::assets::ModelRef,
    terrain_box_model: engine3d::assets::ModelRef,
}

impl<C: Camera> engine3d::Game for Game<C> {
    type StaticData = GameData;
    fn start(engine: &mut Engine) -> (Self, Self::StaticData) {
        use rand::Rng;
        let wall = Wall {
            body: Plane {
                n: Vec3::new(0.0, 1.0, 0.0),
                d: 0.0,
            },
            control: (0, 0),
        };
        let player = Player {
            body: Sphere {
                c: Pos3::new(0.0, 3.0, 0.0),
                r: 0.3,
            },
            velocity: Vec3::zero(),
            acc: Vec3::zero(),
            omega: Vec3::zero(),
            rot: Quat::new(1.0, 0.0, 0.0, 0.0),
            hp:100
        };
        let camera = C::new();
        let mut rng = rand::thread_rng();
        let marbles = Marbles {
            body: (0..NUM_MARBLES)
                .map(move |_x| {
                    let x = rng.gen_range(-5.0..5.0);
                    let y = rng.gen_range(1.0..5.0);
                    let z = rng.gen_range(-5.0..5.0);
                    let r = rng.gen_range(0.1..1.0);
                    Sphere {
                        c: Pos3::new(x, y, z),
                        r,
                    }
                })
                .collect::<Vec<_>>(),
            velocity: vec![Vec3::zero(); NUM_MARBLES],
            hp: vec![5; NUM_MARBLES],
        };

        let mut rng = rand::thread_rng();
        let terrain_boxes = Terrain_Boxes {
            body: (0..NUM_TERRAIN_BOXES)
                .map(move |_x| {
                    let x = rng.gen_range(-5.0..5.0);
                    let y = rng.gen_range(1.0..5.0);
                    let z = rng.gen_range(-5.0..5.0);
                    let x_axis = Vec3 {
                        x: 1.0,
                        y: 0.0,
                        z: 0.0,
                    };
                    let y_axis = Vec3 {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    };
                    let z_axis = Vec3 {
                        x: 0.0,
                        y: 0.0,
                        z: 1.0,
                    };
                    let half_sizes = Vec3 {
                        x: 1.0,
                        y: 1.0,
                        z: 1.0,
                    };
                    Box {
                        c: Pos3::new(x, y, z),
                        axes: Mat3 {
                            x: x_axis,
                            y: y_axis,
                            z: z_axis,
                        },
                        half_sizes,
                    }
                })
                .collect::<Vec<_>>(),
            velocity: vec![Vec3::zero(); NUM_TERRAIN_BOXES],
            hp: vec![5; NUM_TERRAIN_BOXES],
        };

        let wall_model = engine.load_model("floor.obj");
        let marble_model = engine.load_model("sphere.obj");
        let player_model = engine.load_model("capsule.obj");
        let terrain_box_model = engine.load_model("box.obj");
        (
            Self {
                // camera_controller,
                marbles,
                wall,
                player,
                terrain_boxes,
                camera,
                // TODO nice this up somehow
                mm: vec![],
                mw: vec![],
                pm: vec![],
                pw: vec![],
                tw: vec![],
            },
            GameData {
                wall_model,
                marble_model,
                player_model,
                terrain_box_model,
            },
        )
    }
    fn render(
        &mut self,
        rules: &Self::StaticData,
        assets: &engine3d::assets::Assets,
        igs: &mut InstanceGroups,
    ) {
        self.wall.render(rules, igs);
        self.marbles.render(rules, igs);
        self.player.render(rules, igs);
        //self.terrain_boxes.render(rules,igs);
        // self.camera.render(rules, igs);
    }
    fn update(&mut self, _rules: &Self::StaticData, engine: &mut Engine) {
        // dbg!(self.player.body);
        // TODO update player acc with controls
        // TODO update camera with controls/player movement
        // TODO show how spherecasting could work?  camera pseudo-entity collision check?  camera entity for real?
        // self.camera_controller.update(engine);

        self.player.acc = Vec3::zero();
        if engine.events.key_held(KeyCode::W) {
            self.player.acc.z = 1.0;
        } else if engine.events.key_held(KeyCode::S) {
            self.player.acc.z = -1.0;
        }

        if engine.events.key_held(KeyCode::A) {
            self.player.acc.x = 1.0;
        } else if engine.events.key_held(KeyCode::D) {
            self.player.acc.x = -1.0;
        }
        if self.player.acc.magnitude2() > 1.0 {
            self.player.acc = self.player.acc.normalize();
        }

        if engine.events.key_held(KeyCode::Q) {
            self.player.omega = Vec3::unit_y();
        } else if engine.events.key_held(KeyCode::E) {
            self.player.omega = -Vec3::unit_y();
        } else {
            self.player.omega = Vec3::zero();
        }

        if engine.events.key_pressed(KeyCode::Space) {
            //let forward = self.player.rot.v.normalize();
            self.marbles.body.push(Sphere {
                c: Pos3 {
                    x: self.player.body.c.x,
                    y: 0.25,
                    z: self.player.body.c.z + 0.3,
                },
                r: 0.5,
            });
            self.marbles.velocity.push(Vec3 {
                x: 0.0,
                y: 0.0,
                z: 3.0,
            });
            self.marbles.hp.push(5);
        }

        // orbit camera
        self.camera.update(&engine.events, &self.player);

        self.wall.integrate();
        self.player.integrate();
        self.marbles.integrate();
        self.terrain_boxes.integrate();
        self.camera.integrate();

        {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for (body, vel) in self.marbles.iter_mut() {
                if (body.c.distance(Pos3::new(0.0, 0.0, 0.0))) >= 40.0 {
                    body.c = Pos3::new(
                        rng.gen_range(-5.0..5.0),
                        rng.gen_range(1.0..5.0),
                        rng.gen_range(-5.0..5.0),
                    );
                    *vel = Vec3::zero();
                }
            }
        }
        self.mm.clear();
        self.mw.clear();
        self.pm.clear();
        self.pw.clear();
        self.tw.clear();
        let mut pb = [self.player.body];
        let mut pv = [self.player.velocity];
        let mut ph = [self.player.hp];
        collision::gather_contacts_ab(&pb, &self.marbles.body, &mut self.pm);
        collision::gather_contacts_ab(&pb, &[self.wall.body], &mut self.pw);
        collision::gather_contacts_ab(&self.marbles.body, &[self.wall.body], &mut self.mw);
        //collision::gather_contacts_ab(&self.terrain_boxes.body, &[self.wall.body], &mut self.tw);
        collision::gather_contacts_aa(&self.marbles.body, &mut self.mm);
        collision::restitute_dyn_stat(&mut pb, &mut pv, &[self.wall.body], &mut self.pw);
        collision::restitute_dyn_stat(
            &mut self.marbles.body,
            &mut self.marbles.velocity,
            &[self.wall.body],
            &mut self.mw,
        );
        /*collision::restitute_dyn_stat(
            &mut self.terrain_boxes.body,
            &mut self.terrain_boxes.velocity,
            &[self.wall.body],
            &mut self.tw,
        );*/
        let mut marbles_to_remove = vec![];
        let mut player_to_remove = vec![];
        collision::restitute_dyns(
            &mut self.marbles.body,
            &mut self.marbles.velocity,
            &mut self.marbles.hp,
            &mut self.mm,
            &mut marbles_to_remove,
        );
        collision::restitute_dyn_dyn(
            &mut vec![self.player.body],
            &mut vec![self.player.velocity],
            &mut vec![self.player.hp],
            &mut self.marbles.body,
            &mut self.marbles.velocity,
            &mut self.marbles.hp,
            &mut self.pm,
            &mut player_to_remove,
            &mut marbles_to_remove,
        );
        self.player.body = pb[0];
        self.player.velocity = pv[0];

        for collision::Contact { a: ma, .. } in self.mw.iter() {
            // apply "friction" to marbles on the ground
            self.marbles.velocity[*ma] *= 0.995;
        }
        for collision::Contact { a: pa, .. } in self.pw.iter() {
            // apply "friction" to players on the ground
            assert_eq!(*pa, 0);
            self.player.velocity *= 0.98;
        }

        clean(&mut self.marbles.body,&mut marbles_to_remove);
        clean(&mut self.marbles.velocity,&mut marbles_to_remove);
        clean(&mut self.marbles.hp,&mut marbles_to_remove);

        self.camera.update_camera(engine.camera_mut());
    }
}

fn clean<T>(vec: &mut Vec<T>,indices: &mut Vec<usize>){
    indices.sort_by(|a, b| b.cmp(a));
    for &a in indices.iter(){
        vec.remove(a);
    }
}

fn main() {
    env_logger::init();
    let title = env!("CARGO_PKG_NAME");
    let window = winit::window::WindowBuilder::new().with_title(title);
    run::<GameData, Game<OrbitCamera>>(window, std::path::Path::new("content"));
}
