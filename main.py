# import FastAPI
from fastapi import FastAPI, Query

# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# import joblib for model naive bayes
import joblib 

# make instance app
app = FastAPI(
    title="Naive Bayes Model API",
    description="A simple API that use NLP model to predict the title of the book",
    version="0.1",
)

# steaming function with sastrawi
def text_sastrawi(text, remove_stop_words=True, lemmatize_words=True):
    # create stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    # stemming process
    text = stemmer.stem(text)

    return text

# load the nb model
with open("models/judul_model_nb.pkl", "rb") as f:
    model = joblib.load(f)

# load the tfid coverter
with open("models/judul_model_tfid.pkl", "rb") as f:
    vectorizer = joblib.load(f)
    
@app.get("/judul/{judul}", tags=["Items"], summary="Read Items", description="Ini adalah data berbagai macam item")
async def read_item(judul: str = Query(None, min_length=2, max_length=12)):
    
    # clean the review
    judul_stemming = text_sastrawi(judul)

    # tfid
    judul_tfid = vectorizer.transform([judul_stemming])

    # perform prediction
    prediction = model.predict(judul_tfid)
    output = int(prediction)
    probas = model.predict_proba(judul_tfid)

    output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    sentiments = {
        0: "Umum", 
        1: "Filsafat dan Psikologi",
        2: "Agama",
        3: "Sosial",
        4: "Bahasa",
        5: "Sains dan Matematika",
        6: "Teknologi",
        7: "Seni dan Rekreasi",
        8: "Literartur dan Sastra",
        9: "Sejarah dan Geografi"
    }
    
    # show results
    result = {
        "judul_origin": judul,
        "judul_stemming": judul_stemming,
        "prediction": sentiments[output],
        "probability": output_probability,
        "probability_class": {
            "{:.2f}".format(float(probas[:, 0])) : "Umum", 
            "{:.2f}".format(float(probas[:, 1])) : "Filsafat dan Psikologi",
            "{:.2f}".format(float(probas[:, 2])) : "Agama",
            "{:.2f}".format(float(probas[:, 3])) : "Sosial",
            "{:.2f}".format(float(probas[:, 4])) : "Bahasa",
            "{:.2f}".format(float(probas[:, 5])) : "Sains dan Matematika",
            "{:.2f}".format(float(probas[:, 6])) : "Teknologi",
            "{:.2f}".format(float(probas[:, 7])) : "Seni dan Rekreasi",
            "{:.2f}".format(float(probas[:, 8])) : "Literartur dan Sastra",
            "{:.2f}".format(float(probas[:, 9])) : "Sejarah dan Geografi"
        }
    }

    return result

# .\env\Scripts\activate aktifasi modul env py
# uvicorn main:app --reload untuk menyalakan server
# uvicorn main:app --reload untuk hot reload