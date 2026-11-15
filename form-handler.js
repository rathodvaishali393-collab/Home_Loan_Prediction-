document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("loan-form");
  const message = document.getElementById("form-message");

  if (!form) return;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const button = form.querySelector('button[type="submit"]');
    const original = button.innerHTML;
    button.disabled = true;
    button.innerHTML = "⏳ Running ML Prediction...";

    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());
    delete payload.fullName;

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || "Prediction failed.");
      }

      sessionStorage.setItem("loanPrediction", JSON.stringify({
        applicantName: formData.get("fullName"),
        input: payload,
        result
      }));

      window.location.href = "results.html";
    } catch (error) {
      message.style.display = "block";
      message.className = "result-box error";
      message.innerHTML = `<strong>Prediction Error:</strong> ${error.message}<br><small>Make sure the Flask server is running.</small>`;
    } finally {
      button.disabled = false;
      button.innerHTML = original;
    }
  });
});