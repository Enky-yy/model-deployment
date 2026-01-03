from flask import Flask , render_template , request
import pickle

# loading models 
cv= pickle.load(open("Deployment/model-deployment/models/cv.pkl", "rb"))
clf= pickle.load(open("Deployment/model-deployment/models/clf.pkl", "rb"))

app =Flask(__name__)

@app.route("/")
def home():
    
    return render_template("index.html")

@app.route("/predict" , methods=["POST"])
def predict():
    if request.method=="POST":
        email_text = request.form.get("email-content")

    tokenized_email = cv.transform([email_text])
    prediction = clf.predict(tokenized_email)
    prediction =1 if prediction==1 else -1
    return render_template("index.html", prediction=prediction , email_text=email_text)

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)