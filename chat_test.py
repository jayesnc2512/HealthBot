import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk

import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


training = pd.read_csv("Data/Training.csv")
testing = pd.read_csv("Data/Testing.csv")
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training["prognosis"]
y1 = y


reduced_data = training.groupby(training["prognosis"]).max()

# mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)
testx = testing[cols]
testy = testing["prognosis"]
testy = le.transform(testy)

# Before fitting the classifier, set feature names
clf1 = DecisionTreeClassifier()
clf1.feature_names = cols  # Set feature names
clf = clf1.fit(x_train, y_train)

# clf1 = DecisionTreeClassifier()
# clf = clf1.fit(x_train, y_train)
# print(clf.score(x_train,y_train))
# print ("cross result========")
scores = cross_val_score(clf, x_test, y_test, cv=3)
# print (scores)
# print(scores.mean())


model = SVC()
model.fit(x_train, y_train)
# print("for svm: ")
# print(model.score(x_test, y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols


def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty("voice", "english+f5")
    engine.setProperty("rate", 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index


def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + severityDictionary[item]

    if (sum * days) / (len(exp) + 1) > 13:
        return "You should take the consultation from a doctor."
    else:
        return "It might not be that bad, but you should take precautions."


def getDescription():
    global description_list
    with open("MasterData/symptom_Description.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open("MasterData/Symptom_severity.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open("MasterData/symptom_precaution.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(" ", "_")  # Replace spaces with underscores
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []


def sec_predict(symptoms_exp):
    df = pd.read_csv("Data/Training.csv")
    X = df.iloc[:, :-1]
    y = df["prognosis"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=20
    )
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


# Your existing chatbot logic goes here
def chatbot_logic(text_widget, result_text_widget):
    # Process the input data and generate a response
    result_messages = tree_to_code(clf, cols)
    # Display the result messages in the result_text_widget
    for message in result_messages:
        result_text_widget.insert("end", message + "\n")
    result_text_widget.see("end")  # Scroll to the end of the result_text_widget


def tree_to_code(tree, feature_names):
    getSeverityDict()
    getDescription()
    getprecautionDict()

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    def get_input(text):
        return simpledialog.askstring("User Input", text)

    result_messages = []

    while True:
        symptom_input = get_input("Enter the symptom you are experiencing:")
        disease_input = symptom_input.strip()

        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            result_messages.append("Searches related to input:")
            for num, it in enumerate(cnf_dis):
                result_messages.append(f"{num}) {it}")

            if num != 0:
                conf_inp = simpledialog.askinteger(
                    "Select Symptom",
                    f"Select the one you meant (0 - {num}):",
                    initialvalue=0,
                )
            else:
                conf_inp = 0

            disease_input = cnf_dis[conf_inp]
            break
        else:
            result_messages.append("Enter a valid symptom.")

    # num_days
    while True:
        num_days = get_input("Okay. From how many days?")
        try:
            num_days = int(num_days)
            break
        except ValueError:
            result_messages.append("Enter a valid number of days.")

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[
                reduced_data.loc[present_disease].values[0].nonzero()
            ]

            result_messages.append("Are you experiencing any symptoms?")
            symptoms_exp = []
            for syms in list(symptoms_given):
                while True:
                    response = get_input(f"{syms} ? (yes/no)")
                    if response == "yes" or response == "no":
                        break
                    else:
                        result_messages.append("Provide a proper answer (yes/no).")

                if response == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            result_message = calc_condition(symptoms_exp, num_days)

            result_messages.append(result_message)

            if present_disease[0] == second_prediction[0]:
                result_messages.append(f"You may have {present_disease[0]}")
                result_messages.append(description_list[present_disease[0]])
            else:
                result_messages.append(
                    f"You may have {present_disease[0]} or {second_prediction[0]}"
                )
                result_messages.append(description_list[present_disease[0]])
                result_messages.append(description_list[second_prediction[0]])

            precution_list = precautionDictionary[present_disease[0]]
            result_messages.append("Take the following measures:")
            for i, j in enumerate(precution_list):
                result_messages.append(f"{i + 1}) {j}")

    recurse(0, 1)
    return result_messages


class HealthChatBot:
    def __init__(self, root):
        self.root = root
        self.root.title("HealthCare ChatBot")
        self.name = ""

        self.frame = ttk.Frame(self.root)
        self.frame.grid(row=0, column=0, padx=10, pady=10)

        self.label = ttk.Label(self.frame, text="Your Name?")
        self.label.grid(row=0, column=0)

        self.name_entry = ttk.Entry(self.frame, width=40)
        self.name_entry.grid(row=0, column=1)

        self.start_button = ttk.Button(self.frame, text="Start", command=self.get_name)
        self.start_button.grid(row=1, column=0, columnspan=2)

        self.text_widget = tk.Text(self.frame, height=3, width=40)
        self.text_widget.grid(row=2, columnspan=3)

        # Create a separate text widget for displaying the result message
        self.result_text_widget = tk.Text(self.frame, height=10, width=40)
        self.result_text_widget.grid(row=3, columnspan=3)

    def get_name(self):
        self.name = self.name_entry.get()
        self.name_entry.delete(0, "end")

        self.label.config(text=f"Hello, {self.name}")

        # Call your chatbot logic here with self.name and self.text_widget
        chatbot_logic(self.text_widget, self.result_text_widget)
        # self.text_widget.delete("1.0", "end")
        # self.text_widget.insert("end", response)


if __name__ == "__main__":
    root = tk.Tk()
    app = HealthChatBot(root)
    root.mainloop()
