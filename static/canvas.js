const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let drawing = false;
let expression = "";

/* WHITE background */
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

/* BLACK drawing */
ctx.lineWidth = 8;
ctx.lineCap = "round";
ctx.strokeStyle = "black";


canvas.addEventListener("mousedown", () => drawing = true);
canvas.addEventListener("mouseup", () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener("mousemove", draw);

function draw(e) {
    if (!drawing) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
}

function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
}

function predict() {
    const dataURL = canvas.toDataURL("image/png");

    fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({image: dataURL})
    })
    .then(res => res.json())
    .then(data => {
        expression += data.symbol;
        document.getElementById("expression").innerText = expression;
        clearCanvas();
    });
}
