# autocar
A lego car that can drive autonomously on a simple road consisting of black spray on brown paper. To drives autonomously it uses a Convolutional Neural Network that learnt how to drive by takings images from the camera and the position of the servo while a human was controlling the car. These training samples are stored in data/manual_round{1..4} and the number after the underscore is the position of the servo. Track A (the one to the left) is the one where training data was collected and the car was tested on the track B (the one to the right). 

## Result
It did not manage to drive autonomously on the track A where it was trained but could drive on the track B that it wasn't trained on. This could be because the track A has much sharper curves that requires it to turn as much at it possibly can. Track B on the other hand has very smoothe curves.

<img src="https://i.ibb.co/rwK8NCf/cof.jpg" alt="cof" style="width:25%"> <img src="https://i.ibb.co/VWPYnQh/cof.jpg" alt="cof" border="0" style="width:25%"> <img src="https://i.ibb.co/CwVBZbV/cof.jpg" alt="cof" border="0" style="width:30%">
<br /> Lego car, Track A, Track B
