document.getElementById("analyzeBtn").addEventListener("click", function () {

    const textArea = document.getElementById("feedback");
    const text = textArea.value.trim();
    const resultBox = document.querySelector(".result-box");
    const sentimentEl = document.getElementById("sentiment");
    const confidenceEl = document.getElementById("confidence");
    const button = this;

    // Input validation
    if (text.length === 0) {
        alert("Please enter student feedback before analysis.");
        return;
    }

    // UI loading state
    button.disabled = true;
    button.innerText = "Analyzing...";
    resultBox.style.display = "none";

    fetch("/analyze", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ feedback: text })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Server error");
        }
        return response.json();
    })
    .then(data => {
        resultBox.style.display = "block";

        // Reset sentiment classes
        sentimentEl.classList.remove(
            "sentiment-positive",
            "sentiment-neutral",
            "sentiment-negative"
        );

        // Set sentiment text
        sentimentEl.innerText = data.sentiment;
        confidenceEl.innerText = data.confidence + "%";

        // Apply sentiment-based styling
        if (data.sentiment === "Positive") {
            sentimentEl.classList.add("sentiment-positive");
        } else if (data.sentiment === "Neutral") {
            sentimentEl.classList.add("sentiment-neutral");
        } else {
            sentimentEl.classList.add("sentiment-negative");
        }
    })
    .catch(error => {
        alert("An error occurred during analysis. Please try again.");
        console.error(error);
    })
    .finally(() => {
        button.disabled = false;
        button.innerText = "Analyze Feedback";
    });
});
