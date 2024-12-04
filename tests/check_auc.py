import pandas as pd
import re
from sklearn.metrics import roc_auc_score

model_auc=[("-[66] -[261] [506] -[1055] [1314] [1318] -[1515] -[1690]",0.782),
              ("-[170] -[261] [309] -[619] -[827] -[1055] [1314] -[1515]",.916),
              ("-[127] -[261] [309] -[827] -[846] -[1055] [1314] -[1515]",.973),
              ("[606] -[846] -[1055] [1118] [1314] [1330]",.902),
              ("-[185] -[331] [710] -[827] [878] -[1012] -[1055] -[1515]",.867),
              ("-[170] -[204] [223] [566] -[648] [799] -[827] [878] -[1119] [1318] -[1515]",.818 ),
              ("-[170] -[261] [309] -[358] -[619] [786] -[827] -[1055] -[1515]",.893),
              ("-[125] -[185] [736] [888] -[1055] [1318] [1340] -[1515]",.884) ]


X_file="samples/Qin2014/Xtest.tsv"
y_file="samples/Qin2014/Ytest.tsv"

X=pd.read_csv(X_file, sep='\t', index_col=0)
y=pd.read_csv(y_file, sep='\t', index_col=0)


model_token=re.compile(r'(-)?\[(.*)\]')

for model_string, rust_auc in model_auc:
    model = []
    for item in model_string.split(' '):
        sign,index=model_token.match(item).groups()
        model.append((int(index),-1 if sign=='-' else 1))
    def eval_row(r):
        return sum([r.iloc[k-1]*v for (k,v) in model])

    value=X.apply(eval_row, axis=0)
    auc=roc_auc_score(y,value)
    print(model_string, rust_auc, auc)