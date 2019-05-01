var src = document.getElementById("text");

var clientX = 0;

src.addEventListener('touchmove', function(e) {
    if (Math.abs(e.touches[0].clientX - clientX) > 5) {
        var xhttp = new XMLHttpRequest();
        var rect = src.getBoundingClientRect();
        var width = rect.right-rect.left;
        xhttp.open("POST", "/servo", true);
        xhttp.setRequestHeader("Content-Type", "application/json");
        xhttp.send(JSON.stringify((clientX)/width));
        clientX = e.touches[0].clientX;
    }
}, false);

src.addEventListener("touchstart", function(e) {
    var xhttp = new XMLHttpRequest();
    xhttp.open("POST", "/motor", true);
    xhttp.setRequestHeader("Content-Type", "application/json");
    xhttp.send(JSON.stringify(true));
}, false);

src.addEventListener("touchend", function(e) {
    var xhttp = new XMLHttpRequest();
    xhttp.open("POST", "/motor", true);
    xhttp.setRequestHeader("Content-Type", "application/json");
    xhttp.send(JSON.stringify(false));
}, false)