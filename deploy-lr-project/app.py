from flask import Flask,render_template,request
import pickle

app = Flask(__name__)
dec_trees = pickle.load(open('C:/Users/MATO/PythonProjects/deploy-lr/deploy-lr-project/dec_trees.pkl','rb'))

@app.route("/")
def page():
    return render_template('index.html')

@app.route("/prediction",methods=['POST'])
def prediction():
    sex = float(request.form['sex'])
    age = float(request.form['age'])
    hypertension = float(request.form['hypertension'])
    heart_disease=float(request.form['heart_disease'])
    ever_married=float(request.form['marriage'])
    work_type=float(request.form['work'])
    residence_type=float(request.form['live'])
    avg_glucose_level=float(request.form['avg_glucose'])
    bmi=float(request.form['bmi'])
    smoking_status=float(request.form['smoking'])
    
    dec_trees_prediction = dec_trees.predict([[sex,age,hypertension,heart_disease,ever_married,work_type,residence_type,avg_glucose_level,bmi,smoking_status]])
    knn_output = dec_trees_prediction[0]

    if (knn_output == 0):
        return render_template('index.html',prediction_text=f'The possibility of a stroke is low.')
    else: 
        return render_template('index.html', prediction_text = f'The possibility of a stroke is high.')


if __name__ == "__main__":
    app.run()