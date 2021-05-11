use ambisonic::Ambisonic;
use ambisonic::SoundController;
use ambisonic::{rodio, AmbisonicBuilder};
use rodio::{Decoder, OutputStream, Sink, source::{Amplify, Buffered, Delay, Repeat}};
use rodio::source::{SineWave, Source};
//change: added crates
use cgmath::Point3;
use engine3d::{
    collision::{self},
    events::*,
    geom::*,
    render::InstanceGroups,
    lights::Light,
    run, Engine, DT,
};
use rand::{self, Rng, thread_rng};
use std::{f32::consts::PI,usize, io::BufReader, fs::File};
use winit;

const NUM_TERRAIN_BOXES_DYN: usize = 40;
const NUM_TERRAIN_BOXES_STAT: usize = 0;
const ORIGIN: Pos3 = Pos3::new(0.0, 0.0, 0.0);

#[derive(Clone, Debug)]
pub struct Player {
    pub body: Sphere,
    pub velocity: Vec3,
    pub acc: Vec3,
    pub hp: usize,
}

// TODO: implement player info
impl Player {
    const MAX_SPEED: f32 = 3.0;
    fn render(&self, rules: &GameData, igs: &mut InstanceGroups) {
        igs.render(
            rules.player_model,
            engine3d::render::InstanceRaw::new(
                (Mat4::from_translation(self.body.c.to_vec() - Vec3::new(0.0, 0.2, 0.0))
                    * Mat4::from_scale(self.body.r)
                    * Mat4::from(self.body.rot))
                .into(),
            ),
        );
    }
    fn integrate(&mut self) {
        //canceled gravity for player
        self.velocity += ((self.body.rot * self.acc) + Vec3::new(0.0, 0.0, 0.0)) * DT;
        if self.velocity.magnitude() > Self::MAX_SPEED {
            self.velocity = self.velocity.normalize_to(Self::MAX_SPEED);
        }
        self.body.c += self.velocity * DT;
        self.body.rot += 0.5
            * DT
            * Quat::new(0.0, self.body.omega.x, self.body.omega.y, self.body.omega.z)
            * self.body.rot;
    }
}

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
        self.player_rot = player.body.rot;
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
pub struct FixOrbitCamera {
    pub pitch: f32,
    pub yaw: f32,
    pub distance: f32,
    player_pos: Pos3,
    player_rot: Quat,
}

impl Camera for FixOrbitCamera {
    fn new() -> Self {
        Self {
            pitch: 0.3,
            yaw: 0.0,
            distance: 5.0,
            player_pos: Pos3::new(0.0, 0.0, 0.0),
            player_rot: Quat::new(1.0, 0.0, 0.0, 0.0),
        }
    }
    fn update(&mut self, _events: &engine3d::events::Events, player: &Player) {
        /* disable player control over camera
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
        }*/
        self.player_pos = player.body.c;
        self.player_rot = player.body.rot;
        // TODO: when player moves, slightly move backwards from player. Effect maginitude defined here.
        let mut rng = rand::thread_rng();
        if (player.acc.z) > 0.0 {
            self.distance = (self.distance + 0.03).clamp(5.0, 6.0);
            self.pitch = (self.pitch + rng.gen_range(-0.001..0.001)).clamp(0.295, 0.305);
            self.yaw = (self.yaw + rng.gen_range(-0.001..0.001)).clamp(-0.005, 0.005);
        } else {
            self.distance = (self.distance - 0.03).clamp(5.0, 6.0);
            self.pitch = 0.3;
            self.yaw = 0.0;
        }
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
        self.player_rot = player.body.rot;
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
//change:added struct
pub struct Audio
{
    scene: Ambisonic,
    sound: Buffered<Decoder<BufReader<File>>>,
    music: Repeat<Decoder<BufReader<File>>>,
    collision_sound: Option<SoundController>
}

#[derive(Clone, Debug)]
pub struct Marbles {
    pub body: Vec<Sphere>,
    pub velocity: Vec<Vec3>,
    pub acc: Vec<Vec3>,
    pub hp: Vec<usize>,
}

// Ziang: I think we can base our game with marbles & boxes...
impl Marbles {
    fn render(&self, rules: &GameData, igs: &mut InstanceGroups) {
        igs.render_batch(
            rules.marble_model,
            self.body.iter().map(|body| engine3d::render::InstanceRaw::new(
                (Mat4::from_translation(body.c.to_vec())
                    * Mat4::from_scale(body.r)
                    * Mat4::from(body.rot))
                .into()
            )),
        );
    }
    fn integrate(&mut self) {
        for ((body, vel), acc) in self
            .body
            .iter_mut()
            .zip(self.velocity.iter_mut())
            .zip(self.acc.iter())
        {
            // The latest implementation enforces a -GRAVITY y acceleration on all newly created marbles.
            *vel += acc * DT;
            body.c += *vel * DT;
            body.rot +=
                0.5 * DT * Quat::new(0.0, body.omega.x, body.omega.y, body.omega.z) * body.rot;
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
            engine3d::render::InstanceRaw::new(
                (Mat4::from(cgmath::Quaternion::between_vectors(
                    Vec3::new(0.0, 1.0, 0.0),
                    self.body.n,
                )) * Mat4::from_translation(Vec3::new(0.0, -0.025, 0.0))
                    * Mat4::from_nonuniform_scale(0.5, 0.05, 0.5))
                .into(),
            ),
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
pub struct Terrain_Boxes_Stat {
    pub body: Vec<Box>,
    pub velocity: Vec<Vec3>,
    pub hp: Vec<usize>,
}

impl Terrain_Boxes_Stat {
    fn add_scaled_cube(&mut self, pos: Pos3, scale: f32) {
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
        } * scale;
        self.body.push(Box {
            c: pos,
            axes: Mat3 {
                x: x_axis,
                y: y_axis,
                z: z_axis,
            } * scale,
            half_sizes,
            omega: Vec3::unit_y(),
            rot: Quat::new(1.0, 0.0, 0.0, 0.0),
        });
        self.velocity.push(Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
        self.hp.push(1000000);
    }

    fn render(&self, rules: &GameData, igs: &mut InstanceGroups) {
        igs.render_batch(
            rules.terrain_box_stat_model,
            self.body.iter().map(|body| engine3d::render::InstanceRaw::new(
                (Mat4::from_translation(body.c.to_vec())
                    * Mat4::from_scale(body.half_sizes.x)
                    * Mat4::from(body.rot))
                .into(),
            )),
        );
    }
    fn integrate(&mut self) {
        
        for vel in self.velocity.iter_mut() {
            *vel += Vec3::new(0.0, 0.0, 0.0) * DT;
            *vel *= 0.98;
        }
        for (body, vel) in self.body.iter_mut().zip(self.velocity.iter()) {
            //body.c += vel * DT;
            body.rot +=
                0.5 * DT * Quat::new(0.0, body.omega.x, body.omega.y, body.omega.z) * body.rot;
        }
    }
    fn _iter_mut(&mut self) -> impl Iterator<Item = (&mut Box, &mut Vec3)> {
        self.body.iter_mut().zip(self.velocity.iter_mut())
    }
}

#[derive(Clone, Debug)]
pub struct Terrain_Boxes_Dyn {
    pub body: Vec<Box>,
    pub velocity: Vec<Vec3>,
    pub hp: Vec<usize>,
}

impl Terrain_Boxes_Dyn {
    fn add_scaled_cube(&mut self, pos: Pos3, scale: f32) {
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
        } * scale;
        let scale = 0.3_f32;
        let mut rng = thread_rng();
        let omega = Vec3::new(rng.gen_range(0.0..1.0),rng.gen_range(0.0..1.0),rng.gen_range(0.0..1.0));
        self.body.push(Box {
            c: pos,
            axes: Mat3 {
                x: x_axis,
                y: y_axis,
                z: z_axis,
            } * scale,
            half_sizes,
            omega:omega.normalize(),
            rot: Quat::new(1.0, 0.0, 0.0, 0.0),

        });
        self.velocity.push(Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
        self.hp.push(1);
    }

    fn render(&self, rules: &GameData, igs: &mut InstanceGroups) {
        igs.render_batch(
            rules.terrain_box_dyn_model,
            self.body.iter().map(|body| engine3d::render::InstanceRaw::new(
                (Mat4::from_translation(body.c.to_vec())
                    * Mat4::from_scale(body.half_sizes.x)*Mat4::from(body.rot))
                .into(),
            )),
        );
    }
    fn integrate(&mut self) {
        for vel in self.velocity.iter_mut() {
            *vel += Vec3::new(0.0, 0.0, 0.0) * DT;
        }
        for (body, vel) in self.body.iter_mut().zip(self.velocity.iter()) {
            body.c += vel * DT;
            body.rot +=
                0.5 * DT * Quat::new(0.0, body.omega.x, body.omega.y, body.omega.z) * body.rot;
        }
    }
    fn _iter_mut(&mut self) -> impl Iterator<Item = (&mut Box, &mut Vec3)> {
        self.body.iter_mut().zip(self.velocity.iter_mut())
    }
}


struct Game<Cam: Camera> {
    marbles: Marbles,
    wall: Wall,
    terrain_boxes_stat: Terrain_Boxes_Stat,
    terrain_boxes_dyn: Terrain_Boxes_Dyn,
    player: Player,
    camera: Cam,
    pm: Vec<collision::Contact<usize>>,
    pw: Vec<collision::Contact<usize>>,
    mm: Vec<collision::Contact<usize>>,
    mw: Vec<collision::Contact<usize>>,
    tw: Vec<collision::Contact<usize>>,
    tm: Vec<collision::Contact<usize>>,
    tp: Vec<collision::Contact<usize>>,
    dp: Vec<collision::Contact<usize>>,
    dm: Vec<collision::Contact<usize>>,
    score:usize,
    //change:added to game struct
    _soundstream: (rodio::OutputStream, rodio::OutputStreamHandle),
    audio: Audio
}
struct GameData {
    marble_model: engine3d::assets::ModelRef,
    wall_model: engine3d::assets::ModelRef,
    player_model: engine3d::assets::ModelRef,
    terrain_box_stat_model: engine3d::assets::ModelRef,
    terrain_box_dyn_model: engine3d::assets::ModelRef,
}

impl<C: Camera> engine3d::Game for Game<C> {
    type StaticData = GameData;
    fn start(engine: &mut Engine) -> (Self, Self::StaticData) {
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
                omega: Vec3::zero(),
                rot: Quat::new(1.0, 0.0, 0.0, 0.0),
            },
            velocity: Vec3::zero(),
            acc: Vec3::zero(),
            hp: 100,
        };
        let camera = C::new();
        let mut rng = rand::thread_rng();
        let mut marbles = Marbles {
            body: vec![],
            velocity: vec![],
            hp: vec![],
            acc: vec![],
        };

        let mut rng = rand::thread_rng();
        let mut terrain_boxes_stat = Terrain_Boxes_Stat {
            body: vec![],
            velocity: vec![],
            hp: vec![],
        };
        /* 
        for i in 0..NUM_TERRAIN_BOXES_STAT {
            let scale = 0.3 as f32;
            let x = rng.gen_range(1.0..3.0);
            let y = rng.gen_range(1.0..3.0);
            let z = rng.gen_range(1.0..3.0);
            let pos_1 = Pos3::new(x, y, z);
            terrain_boxes_stat.add_scaled_cube(pos_1, scale);
        }*/
        let mut terrain_boxes_dyn = Terrain_Boxes_Dyn{
            body:vec![],
            velocity: vec![],
            hp: vec![],
        };
        for _i in 0..NUM_TERRAIN_BOXES_DYN {
            let scale = 0.3_f32;
            let x = rng.gen_range(-10.0..10.0);
            let y = rng.gen_range(5.0..25.0);
            let z = rng.gen_range(-10.0..10.0);
            let pos_1 = Pos3::new(x, y, z);
            terrain_boxes_dyn.add_scaled_cube(pos_1, scale);
        }

        let wall_model = engine.load_model("floor.obj");
        let marble_model = engine.load_model("sphere.obj");
        let player_model = engine.load_model("capsule.obj");
        let terrain_box_stat_model = engine.load_model("box.obj");
        let terrain_box_dyn_model = engine.load_model("box_dyn.obj");
        engine.set_lights(vec![Light::point(
            Pos3::new(0.0, 10.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
        )]);
        engine.set_ambient(0.1);

        let scene = AmbisonicBuilder::new().build();
        let file = BufReader::new(File::open("content/collide.mp3").unwrap());
        let sound = rodio::Decoder::new(file).unwrap().buffered();
        let music_file = BufReader::new(File::open("content/bgm.ogg").unwrap());
        let music = Decoder::new(music_file).unwrap().repeat_infinite();
        let soundstream = OutputStream::try_default().unwrap();
        soundstream.1.play_raw(music.clone().amplify(0.5).convert_samples()).unwrap();
        (
            Self {
                // camera_controller,
                marbles,
                wall,
                player,
                terrain_boxes_stat,
                terrain_boxes_dyn,
                camera,
                // TODO nice this up somehow
                mm: vec![],
                mw: vec![],
                pm: vec![],
                pw: vec![],
                tw: vec![],
                tm: vec![],
                tp: vec![],
                dp: vec![],
                dm: vec![],
                score:0,
                //change:added to start()
                _soundstream: soundstream,
                audio: Audio {
                    scene,
                    sound,
                    music,
                    collision_sound: None
                }
            },
            GameData {
                wall_model,
                marble_model,
                player_model,
                terrain_box_stat_model,
                terrain_box_dyn_model
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
        self.terrain_boxes_stat.render(rules, igs);
        self.terrain_boxes_dyn.render(rules, igs);
        // self.camera.render(rules, igs);
    }

    fn update(&mut self, _rules: &Self::StaticData, engine: &mut Engine) {
        // dbg!(self.player.body);
        // TODO update player acc with controls
        // TODO update camera with controls/player movement
        // TODO show how spherecasting could work?  camera pseudo-entity collision check?  camera entity for real?
        // self.camera_controller.update(engine);

        // player control over position goes here
        self.player.acc = Vec3::zero();
        if engine.events.key_held(KeyCode::W) {
            self.player.acc.z = 3.0;
        } else if engine.events.key_held(KeyCode::S) {
            self.player.acc.z = -3.0;
        }
        if engine.events.key_held(KeyCode::A) {
            self.player.acc.x = 2.0;
        } else if engine.events.key_held(KeyCode::D) {
            self.player.acc.x = -2.0;
        }
        if engine.events.key_held(KeyCode::Up) {
            self.player.acc.y = 3.0;
        } else if engine.events.key_held(KeyCode::Down) {
            self.player.acc.y = -3.0;
        } else{
            self.player.velocity *= 0.98;
        }
        if self.player.acc.magnitude2() > 27.0 {
            self.player.acc = self.player.acc.normalize_to(4.0);
        }

        // player control over direction goes here
        if engine.events.key_held(KeyCode::Q) {
            self.player.body.omega = Vec3::unit_y();
        } else if engine.events.key_held(KeyCode::E) {
            self.player.body.omega = -Vec3::unit_y();
        } else {
            self.player.body.omega = Vec3::zero();
        }

        if engine.events.key_pressed(KeyCode::Space) {
            // A unit vector that points to uphead uphigh from facing.
            // shooting direction and velocity defined here.
            let forward = (self.player.body.rot
                * Vec3 {
                    x: 0.0,
                    y: 0.2,
                    z: 1.0,
                })
            .normalize();
            self.marbles.body.push(Sphere {
                c: Pos3 {
                    x: self.player.body.c.x + forward.x * 0.3,
                    y: self.player.body.c.y + forward.y * 0.3,
                    z: self.player.body.c.z + forward.z * 0.3,
                },
                r: 0.2,
                omega: Vec3::zero(),
                rot: Quat::new(1.0, 0.0, 0.0, 0.0),
            });
            self.marbles.velocity.push(forward.normalize_to(15.0));
            self.marbles.hp.push(1);
            self.marbles.acc.push(Vec3 {
                x: 0.0,
                y: -9.8,
                z: 0.0,
            });
        }

        // orbit camera
        self.camera.update(&engine.events, &self.player);

        self.wall.integrate();
        self.player.integrate();
        self.marbles.integrate();
        self.terrain_boxes_stat.integrate();
        self.terrain_boxes_dyn.integrate();
        self.camera.integrate();

        self.mm.clear();
        self.mw.clear();
        self.pm.clear();
        self.pw.clear();
        self.tw.clear();
        self.tm.clear();
        self.dp.clear();
        self.dm.clear();
        self.tp.clear();
        let mut pb = [self.player.body];
        let mut pv = [self.player.velocity];
        let mut ph = [self.player.hp];
        let mut marbles_to_remove = vec![];
        let mut player_to_remove = vec![];
        let mut terrains_to_remove = vec![];
        let mut useless = vec![];

        collision::gather_contacts_ab(&pb, &self.marbles.body, &mut self.pm);//
        collision::gather_contacts_ab(&pb, &[self.wall.body], &mut self.pw);//
        collision::gather_contacts_ab(&self.terrain_boxes_stat.body, &[self.wall.body], &mut self.tw);//
        collision::gather_contacts_ab(&self.terrain_boxes_dyn.body, &pb, &mut self.dp);//
        collision::gather_contacts_ab(&self.terrain_boxes_stat.body, &pb, &mut self.tp);//
        collision::gather_contacts_ab(&self.marbles.body, &[self.wall.body], &mut self.mw);//
        collision::gather_contacts_aa(&self.marbles.body, &mut self.mm);//
        collision::gather_contacts_ab(&self.terrain_boxes_stat.body, &self.marbles.body, &mut self.tm);//
        collision::gather_contacts_ab(&self.terrain_boxes_dyn.body, &self.marbles.body, &mut self.dm);//

        collision::restitute_dyn_stat(&mut pb, &mut pv, &[self.wall.body], &mut self.pw);//
        collision::restitute_dyn_stat(//
            &mut self.marbles.body,
            &mut self.marbles.velocity,
            &[self.wall.body],
            &mut self.mw,
        );
        collision::restitute_dyn_stat(//
            &mut self.terrain_boxes_stat.body,
            &mut self.terrain_boxes_stat.velocity,
            &[self.wall.body],
            &mut self.tw,
        );
        collision::restitute_dyns(//
            &mut self.marbles.body,
            &mut self.marbles.velocity,
            &mut self.marbles.hp,
            &mut self.mm,
            &mut marbles_to_remove,
        );
        collision::restitute_dyn_dyn(//
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
        collision::restitute_dyn_dyn(//
            &mut self.terrain_boxes_stat.body,
            &mut self.terrain_boxes_stat.velocity,
            &mut self.terrain_boxes_stat.hp,
            &mut self.marbles.body,
            &mut self.marbles.velocity,
            &mut self.marbles.hp,
            &mut self.tm,
            &mut useless,
            &mut marbles_to_remove,
        );
        collision::restitute_dyn_dyn(//
            &mut self.terrain_boxes_stat.body,
            &mut self.terrain_boxes_stat.velocity,
            &mut self.terrain_boxes_stat.hp,
            &mut vec![self.player.body],
            &mut vec![self.player.velocity],
            &mut vec![self.player.hp],
            &mut self.tp,
            &mut useless,
            &mut player_to_remove,
        );
        collision::restitute_dyn_dyn(//
            &mut self.terrain_boxes_dyn.body,
            &mut self.terrain_boxes_dyn.velocity,
            &mut self.terrain_boxes_dyn.hp,
            &mut vec![self.player.body],
            &mut vec![self.player.velocity],
            &mut vec![self.player.hp],
            &mut self.dp,
            &mut useless,
            &mut player_to_remove,
        );
        collision::restitute_dyn_dyn(//
            &mut self.terrain_boxes_dyn.body,
            &mut self.terrain_boxes_dyn.velocity,
            &mut self.terrain_boxes_dyn.hp,
            &mut self.marbles.body,
            &mut self.marbles.velocity,
            &mut self.marbles.hp,
            &mut self.dm,
            &mut terrains_to_remove,
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
        for collision::Contact { a: ma, .. } in self.mw.iter() {
            // apply "friction" to marbles on the ground
            self.marbles.velocity[*ma] *= 0.995;
        }

        if !self.dm.is_empty() {
            //change:collision sound added
            //checking if there are colliding pairs, then play sound according to boxes' pos
            //really not sure if I grabed the pos right, just trying to get the first box in the colliding pair
            for collision in &self.dm {
                let box_c = self.terrain_boxes_dyn.body[collision.a].c;
                let box_posn = [box_c.x, box_c.y, box_c.z]; 
                
                self
                    .audio
                    .scene
                    .play_at(self.audio.sound.clone().convert_samples(), box_posn);
            }
        }
       
        /*Removal Segment*/
        {
            let mut i=0;
            for (body, vel) in self.marbles.iter_mut() {
                if (body.c.distance(ORIGIN)) >= 40.0 {
                    marbles_to_remove.push(i);
                    i += 1;
                }
            }
            if self.player.body.c.distance(ORIGIN) >= 40.0 {
                if self.player.hp >= 1 {
                    self.player.hp -= 1;
                } else{
                    self.player.hp = 0;
                }
            }
        }
        clean(&mut self.marbles.body, &mut marbles_to_remove);
        clean(&mut self.marbles.velocity, &mut marbles_to_remove);
        clean(&mut self.marbles.hp, &mut marbles_to_remove);
        clean(&mut self.terrain_boxes_dyn.body, &mut terrains_to_remove);
        clean(&mut self.terrain_boxes_dyn.velocity, &mut terrains_to_remove);
        clean(&mut self.terrain_boxes_dyn.hp, &mut terrains_to_remove);

        self.camera.update_camera(engine.camera_mut());
        self.score = NUM_TERRAIN_BOXES_DYN - self.terrain_boxes_dyn.body.len();
    }

    fn is_over(&mut self)->(bool,bool) {
        if self.score == NUM_TERRAIN_BOXES_DYN {
            (true,true)
        } else if self.player.hp == 0 {
            (true,false)
        } else{
            (false,false)
        }
    }
}

fn clean<T>(vec: &mut Vec<T>, indices: &mut Vec<usize>) {
    indices.sort_by(|a, b| b.cmp(a));
    for &a in indices.iter() {
        vec.remove(a);
    }
}

fn main() {
    env_logger::init();
    let title = env!("CARGO_PKG_NAME");
    let window = winit::window::WindowBuilder::new().with_title(title);
    run::<GameData, Game<FixOrbitCamera>>(window, std::path::Path::new("content"));
}