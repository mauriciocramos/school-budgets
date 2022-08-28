# Custom modules
import sys
sys.path.insert(1, '../machine-learning')
from model_selection.multilabel import multilabel_sample_dataframe, multilabel_train_test_split

# Standard modules
import pickle
import numpy as np
import pandas as pd

# Prepare data
df = pd.read_csv('/data/drivendata/TrainingData.csv', index_col=0)
LABELS = ['Function', 'Object_Type', 'Operating_Status', 'Position_Type', 'Pre_K', 'Reporting', 'Sharing', 'Student_Type', 'Use']
NON_LABELS = list(set(df.columns) - set(LABELS))
df[LABELS] = df[LABELS].apply(lambda x: x.astype('category'), axis=0)
y = pd.get_dummies(df[LABELS], prefix_sep='__')


results = pd.DataFrame(columns=['sample_size', 'sample_min_count', 'sample_lessfreq', 'sample_ratio',
                                'test_size', 'test_min_count', 'train_lessfreq', 'test_lessfreq', 'train_ratio',
                                'test_ratio'])
y_mean = y.mean()
# sample min_count sample limited by the population less frequent class
sample_min_count_space = range(0, y.sum().min() + 1)
# sample size space from 0.01 to whole dataset
sample_size_space = np.concatenate([np.linspace(0.01, 0.09, 9), np.linspace(0.1, 0.9, 9), [df.shape[0]]])

for sample_size in sample_size_space: # 19 iterations
    
    for sample_min_count in sample_min_count_space: # 29 iterations
        
        sample_df = multilabel_sample_dataframe(df, y, size=sample_size, min_count=sample_min_count, seed=1)
        dummy_labels = pd.get_dummies(sample_df[LABELS], prefix_sep='__')
        sample_ratio = (dummy_labels.mean() / y_mean).mean()
        # The sample less frequent class upper limits the split min_count otherwise it fails
        sample_less_frequent_class = dummy_labels.sum().min()
        
        
        for test_size in np.linspace(0.1, 0.9, 9):
            
            test_min_count_space = range(0, sample_less_frequent_class + 1)
            
            for test_min_count in test_min_count_space:
                
                The test_size lower limits the test_min_count
                if dummy_labels.shape[1] * test_min_count > np.floor(dummy_labels.shape[0] * test_size):
                    continue
                    
                X_train, X_test, y_train, y_test = multilabel_train_test_split(sample_df[NON_LABELS],
                                                                               dummy_labels,
                                                                               size=test_size,
                                                                               min_count=test_min_count,
                                                                               seed=1)
                train_ratio = (y_train.mean() / y_mean).mean()
                test_ratio = (y_test.mean() / y_mean).mean()
                train_lessfreq = y_train.sum().min()
                test_lessfreq = y_test.sum().min()
                msg = 'SAMPLE {:.2f} min {} lessfreq {} ratio {:.5f} SPLIT {:.1f} min {} LESSFREQ train {} test {} ' \
                      'RATIOS train {:.5f} test {:.5f}'
                print(msg.format(sample_size, sample_min_count, sample_less_frequent_class, sample_ratio, test_size,
                                 test_min_count, train_lessfreq, test_lessfreq, train_ratio, test_ratio))
                results = pd.concat([results,
                                     pd.DataFrame({'sample_size': [sample_size],
                                                   'sample_min_count': [sample_min_count],
                                                   'sample_lessfreq': [sample_less_frequent_class],
                                                   'sample_ratio': [sample_ratio],
                                                   'test_size': [test_size],
                                                   'test_min_count': [test_min_count],
                                                   'train_lessfreq': [train_lessfreq],
                                                   'test_lessfreq': [test_lessfreq],
                                                   'train_ratio': [train_ratio],
                                                   'test_ratio': [test_ratio]})], ignore_index=True)

                
# save generated dataframe
with open('/data/drivendata/optimize-sample-split.pkl', 'wb') as file:
    pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)
