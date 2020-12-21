
use std::path::{Path, PathBuf};
use std::error::Error;
use opencv::imgcodecs::imread;

use rusty_machine::linalg::{Vector, Matrix};

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

// reads a filename in the following format *_13.png where 13 is the servo position of that image.
fn get_servo_pos(path: &Path) -> u32 {
    let file_name = path.file_stem().unwrap().to_str().unwrap();
    let underscore_idx = file_name.find('_').unwrap();
    file_name[underscore_idx+1..].parse::<u32>().unwrap()
}

// converts all the images into a matrix where each row represents an image.
pub fn imgs_to_matrix(imgs: &Vec<opencv::core::Mat>) -> Matrix<f64> {
    let mut data = Vec::<f64>::with_capacity(imgs.len()*(imgs[0].data().unwrap() as &[f64]).len());
    for img in imgs {
       for elem in img.data().unwrap() {
           data.push(*elem);
        }
    }
    let matrix = Matrix::new(imgs.len(), (imgs[0].data().unwrap() as &[f64]).len(), data);
    //matrix
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

    #[test]
    fn test_imgs_to_matrix() {
        let v = vec![
            opencv::core::Mat::from_slice_2d(&vec![[1,2],[3,4]]),
            opencv::core::Mat::from_slice_2d(&vec![[5,6],[7,8]])
        ];
        let mat = imgs_to_matrix(&v);
        //assert_eq!(mat, Matrix::new(2, 4, Vec))

    }
}