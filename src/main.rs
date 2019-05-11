#![feature(proc_macro_hygiene, decl_macro)]
#[macro_use] extern crate rocket;
extern crate rocket_contrib;
extern crate serde_derive;
extern crate serial;
extern crate bufstream;
extern crate opencv;
extern crate rusty_machine;
#[macro_use] extern crate lazy_static;
extern crate rand;

use serial::prelude::*;
use serial::unix::TTYPort;
use std::time::Duration;
use rocket::response::NamedFile;
use rocket_contrib::json::Json;
use std::path::PathBuf;
use std::io::{Write};
use std::sync::Mutex;
use bufstream::BufStream;

use opencv::highgui::*;
use opencv::imgcodecs::*;
use opencv::videoio::VideoCapture;
use opencv::imgproc;

use rand::prelude::*;

use std::marker::Send;
use std::thread;

mod trainer;

struct CarTracker {
    servo_pos: u8,
    cap: SendVideoCapture,
    imgs_dst: Option<String>,
    driving: bool,
}

struct SendVideoCapture {vc: VideoCapture}
unsafe impl Send for SendVideoCapture {}

const SETTINGS: serial::PortSettings = serial::PortSettings {
    baud_rate: serial::Baud9600,
    char_size: serial::Bits8,
    flow_control: serial::FlowNone,
    parity: serial::ParityNone,
    stop_bits: serial::Stop1,
};

lazy_static! {
    static ref SERPORT: Mutex<BufStream<TTYPort>> = { // send [1,15] for the position and [16,17] for stop and start driving respectively.
        let mut port = match serial::open("/dev/ttyACM0") {
            Ok(f) => f,
            Err(e) => match serial::open("/dev/ttyACM1") {
                Ok(inner_f) => inner_f,
                Err(inner_err) => panic!("{:?}, {:?}", e, inner_err),
            },
        };
        port.configure(&SETTINGS).unwrap();
        Mutex::new(BufStream::new(port))
    };
    static ref HIGHGUI_LOCK: Mutex<()> = Mutex::new(());
    static ref TRACKER: Mutex<CarTracker> = Mutex::new(CarTracker::new(0, std::env::args().nth(1)));
}

impl CarTracker {
    fn new(device: i32, dst: Option<String>) -> Self {
        CarTracker {
            servo_pos: 8,
            cap: SendVideoCapture {vc: VideoCapture::index(device).unwrap()},
            imgs_dst: dst,
            driving: false,
        }
    }
    fn update(&mut self) -> Result<(), opencv::Error> {
        let mut img = opencv::mat();
        for _i in 0..5 {
            self.cap.vc.grab()?;
        }
        self.cap.vc.retrieve(&mut img, 0)?;
        let mut smaller_img = opencv::mat();
        imgproc::resize(&img, &mut smaller_img, opencv::core::Size {width:30, height:30}, 0.0, 0.0, 0)?;
        let mut greyscale = opencv::mat();
        imgproc::cvt_color(&smaller_img, &mut greyscale, imgproc::CV_BGR2GRAY, 0)?;
        let param = opencv::types::VectorOfint::new();

        if let Some(dst) = &self.imgs_dst {
            if self.driving {
                let r = random::<u64>();
                let path = format!("{}{:X}_{}.png", dst, r, self.servo_pos);
                imwrite(&path, &greyscale, &param)?;
            }
        }
        
        #[cfg(debug_assertions)]
        {
            let _lock = HIGHGUI_LOCK.lock();
            imshow("kamera", &greyscale)?;
            wait_key(1)?;
        }
        
        Ok(())
    }
}

#[post("/servo", format="application/json", data="<pos>")]
fn pos(pos: Json<f32>) {
    let pos = pos.into_inner();
    println!("{}", pos*14.0);
    //let pos = ((pos*14.0).round()) as u8 + 1;
    let pos = if (pos*14.0) < 0.5 {
        (pos*14.0).round() as u8 + 1
    } else {
        (pos*14.0-0.3).round() as u8 + 1
    };
    let pos = if pos > 15 {15} else {pos};
    {
        let mut port = SERPORT.lock().expect("pos(): Failed to lock SERPORT");
        let mut s = pos.to_string();
        s.push_str("\x0d\x0a");
        (*port).write_all(s.as_bytes()).unwrap();
        port.flush().unwrap();
        println!("{:?}", pos);
    }

    {
        let mut tracker = TRACKER.lock().unwrap();
        tracker.servo_pos = pos;
    }
}

#[post("/motor", format="application/json", data="<mode>")]
fn motor(mode: Json<bool>) {
    let mode = mode.into_inner();
    println!("{:?}", mode);
    let msg = if mode {17} else {16};
    let mut port = SERPORT.lock().unwrap();
    let mut msg = msg.to_string();
    msg.push_str("\x0d\x0a");
    (*port).write_all(msg.as_bytes()).unwrap();
    port.flush().unwrap();
    println!("{:?}", msg);

    {
        let mut tracker = TRACKER.lock().unwrap();
        tracker.driving = mode;
    }
}

#[get("/")]
fn index() -> Option<NamedFile> {
    NamedFile::open("index.html").ok()
}

#[get("/<file..>")]
fn stuff(file: PathBuf) -> Option<NamedFile> {
    NamedFile::open(file).ok()
}

fn main() {
    
    {   
        let mut port = SERPORT.lock().unwrap();
        let mut msg = String::from("18");
        msg.push_str("\x0d\x0a");
        (*port).write_all(msg.as_bytes()).unwrap();
        port.flush().unwrap();
    }
    
    #[cfg(debug_assertions)]
    {
        if let Err(e) = named_window("kamera", WINDOW_NORMAL) {
            panic!("{:?}", e);
        }

        resize_window("kamera", opencv::core::Size::new(600,600)).unwrap();
    }
    
    thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_millis(250));
            let mut tracker = TRACKER.lock().expect("Failed to lock");
            if let Err(e) = tracker.update() {
                eprintln!("{:?}", e);
            }
        }
    });

    rocket::ignite().mount("/", routes![stuff, index, pos, motor]).launch();
}