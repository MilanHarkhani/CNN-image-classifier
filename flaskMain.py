
import os
import operator
from flask import Flask, render_template, request
from predict import Predict_


app = Flask(__name__)




UPLOAD_FOLDER = os.path.basename('static')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/')

def upload_image():

    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload_file():
    global pred
    file = request.files['image']

    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    
    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(f)
    pred_result,results, cls = pred.predict(f)

    result_list =results.ravel().tolist()
    
    result_dict ={prob:cls[i] for i,prob in enumerate(result_list) }
    
    sorted_result = sorted(result_dict.items(), key=operator.itemgetter(1),reverse=False)
    print(sorted_result)
    
    result_str="probability of flowers: \n"
    
    for j in sorted_result:
        result_str += "{} : {}%\n".format(j[1],round(j[0]*100,2))
    return render_template('result.html',user_image=f,name=pred_result,prob =result_str )

if __name__ == '__main__':
    pred = Predict_()
    app.run()