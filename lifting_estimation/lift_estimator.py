import os
import numpy
import pandas
import progressbar
import argparse
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.externals import joblib

MODEL_FILENAME = 'trained_estimator.pkl'


class LiftEstimator(object):
    def __init__(self, model=None, training_df=None):
        self.model = model
        self.training_df = training_df
        self.split_data = None
        self.categories_labels = {}

    def train(self, train_df=None):
        self.split_data = self.get_split_df(train_df)
        if self.model is None:
            self.model = MLPRegressor(solver='adam', alpha=1e-5,
                                      hidden_layer_sizes=(5, 4),
                                      random_state=1,
                                      # learning_rate='adaptive',
                                      learning_rate_init=0.002,
                                      # max_iter=250,
                                      warm_start=True)
        steps = 100
        print('Training...{} steps'.format(steps))
        bar = progressbar.ProgressBar(max_value=steps)
        bar.update(0)
        for i in range(steps):
            self.model.fit(self.split_data['x_train'],
                           self.split_data['y_train'])
            bar.update(i)
        print(' Finished Training')

    def get_input_data(self):
        sex = self.categories_labels['Sex'].transform([
            str(input('your gender (M/F)? '))])
        equipment = str(input('your using equipment?{} '.format(list(
            self.categories_labels['Equipment'].classes_))))
        equipment = self.categories_labels['Equipment'].transform([equipment])
        input_row = \
            {'BodyweightKg': int(input('your body weight? ')),
             'BestDeadliftKg': int(input('your max dead lift? ')),
             'Age': int(input('your age? ')),
             'Sex': sex[0],
             'Equipment': equipment[0]}
        return pandas.DataFrame([input_row])

    def predict(self, input_data=None):
        input_data = input_data if input_data is not \
                                   None else self.split_data['x_test']

        score = self.model.score(self.split_data['x_test'],
                                 self.split_data['y_test'])
        print('score={}'.format(score))
        return self.model.predict(input_data)

    def get_split_df(self, train_df):
        print('Preparing Data...')
        bar = progressbar.ProgressBar(max_value=5)
        bar.update(0)
        training_df = self.drop_unnecessary_columns(train_df)
        bar.update(1)
        training_df = self.fillna(training_df)
        x_data = training_df.drop(['BestBenchKg', 'BestSquatKg'], axis=1)
        bar.update(2)
        x_data = self.set_categorical_data(x_data)
        bar.update(3)
        bar.update(4)
        labels_list = list(tuple(zip(training_df['BestBenchKg'],
                                     training_df['BestSquatKg'])))
        labels = numpy.array(labels_list)
        X_train, X_test, y_train, y_test = train_test_split(
            x_data, labels, test_size=0.33,
            random_state=42)
        bar.update(5)
        progressbar.streams.flush()
        print(' Data is ready')
        return {'x_train': X_train,
                'x_test': X_test,
                'y_train': y_train,
                'y_test': y_test}

    def fillna(self, training_df):
        values = {'BestBenchKg': training_df['BestBenchKg'].mean(),
                  'BestSquatKg': training_df['BestSquatKg'].mean(),
                  'BodyweightKg': training_df['BodyweightKg'].mean(),
                  'BestDeadliftKg': training_df['BestDeadliftKg'].mean(),
                  'Age': training_df['Age'].mean(),
                  'Sex': 'M',
                  'Equipment': training_df['Equipment'].sample(1).iloc[0]}
        training_df = training_df.fillna(value=values)
        return training_df

    def set_categorical_data(self, x_data):
        equipment_encode = preprocessing.LabelEncoder()
        sex_encode = preprocessing.LabelEncoder()
        equipment_encode.fit(x_data['Equipment'])
        sex_encode.fit(x_data['Sex'])
        self.categories_labels = {
            'Sex': sex_encode,
            'Equipment': equipment_encode
        }
        x_data['Equipment'] = equipment_encode.transform(x_data['Equipment'])
        x_data['Sex'] = sex_encode.transform(x_data['Sex'])
        return x_data

    def drop_unnecessary_columns(self, train_df):
        training_df = train_df if train_df is not None else self.training_df
        training_df = training_df[~training_df['Place'].isin(['DD', 'DQ',
                                                              'G', 'NS'])]
        drop_columns = ['MeetID', 'Name', 'Division', 'Squat4Kg', 'Bench4Kg',
                        'Deadlift4Kg', 'TotalKg', 'WeightClassKg',
                        'Place', 'Wilks']
        training_df = training_df.drop(drop_columns, axis=1)
        return training_df

    def save(self):
        joblib.dump(self, MODEL_FILENAME)

    def load(self):
        if os.path.exists(MODEL_FILENAME):
            lift = joblib.load(MODEL_FILENAME)
            self.model = lift.model
            self.split_data = lift.split_data
            self.categories_labels = lift.categories_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", help="make prediction from "
                                           "trained model",
                        action="store_true")
    args = parser.parse_args()

    LIFTING_DATA = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../data/openpowerlifting.csv'))
    training_data = pandas.read_csv(LIFTING_DATA)
    estimator = LiftEstimator()
    if args.estimate:
        estimator.load()
        predict_data = estimator.get_input_data()
        print(estimator.predict(predict_data))
    else:
        estimator.train(train_df=training_data)
        print(estimator.predict())
        estimator.save()

