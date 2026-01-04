import pickle

cv= pickle.load(open("Deployment/model-deployment/models/cv.pkl", "rb"))
clf= pickle.load(open("Deployment/model-deployment/models/clf.pkl", "rb"))

def make_preds(email_text):
    tokenized_email = cv.transform([email_text])
    prediction = clf.predict(tokenized_email)
    prediction =1 if prediction==1 else -1
    return prediction