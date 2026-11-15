from django.shortcuts import render
from .ml_model import predict_loan


def home(request):
    return render(request, "index.html")


def prediction(request):

    if request.method == "POST":

        result = predict_loan(request.POST)

        applicant_name = request.POST.get("fullName", "Applicant")

        probability = (
            result["probability_yes"]
            if result["prediction"] == "Y"
            else result["probability_no"]
        )

        probability_percent = round(probability * 100, 2)

        context = {
            "result": result,
            "applicant_name": applicant_name,
            "input": request.POST,
            "probability_percent": probability_percent,
        }

        return render(request, "results.html", context)

    return render(request, "eligibility-check.html")

def result(request):
    return render(request, "results.html")


def about(request):
    return render(request, "about.html")