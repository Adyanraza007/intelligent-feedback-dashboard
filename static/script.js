document.getElementById("analyzeBtn").addEventListener("click", function() {

    let text = document.getElementById("feedback").value;

    fetch("/analyze", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ feedback: text })
    })
    .then(response => response.json())
    .then(data => {
        document.querySelector(".result-box").style.display = "block";
        document.getElementById("sentiment").innerText = data.sentiment;
        document.getElementById("confidence").innerText = data.confidence + "%";
    });
});
