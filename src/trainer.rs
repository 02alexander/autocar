
use std::path::{Path, PathBuf};
use std::error::Error;
use opencv::imgcodecs::imread;

use rusty_machine::linalg::Vector;

// reads all images in directory and returns a Vec<> the images in opencv mat format
// The Vec<u32> is the servo position of all images.
pub fn load_imgs(dir: PathBuf) -> Result<(Vec<opencv::core::Mat>, Vec<u32>), String> {
    let path = Path::new(&dir);
    let entries =  match path.read_dir() {
        Ok(entries) => {
            entries     
        },
        Err(e) => {
            return Err(format!("Error: {}", e));
        },
    };

    let mut images = Vec::new();
    let mut positions = Vec::new();
    
    for entry in entries {
        if let Ok(entry) = entry {
            positions.push(get_servo_pos(&entry.path()));
            images.push(match imread(entry.path().to_str().unwrap(), 0) {
                Ok(m) => m,
                Err(e) => {return Err(format!("Error: {}", e))},
            });
        }
    }

    Ok((images, positions))
}

fn get_servo_pos(path: &Path) -> u32 {
    let file_name = path.file_stem().unwrap().to_str().unwrap();
    let underscore_idx = file_name.find('_').unwrap();
    file_name[underscore_idx+1..].parse::<u32>().unwrap()
}

pub fn img_to_rulinalg_vec(img: opencv::core::Mat) -> Vector<f64> {

    unimplemented!()
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_get_servo_pos() {
        let path = PathBuf::from("/hejsan/sdf434_34.png");
        assert_eq!(34, get_servo_pos(&path));
    }
}