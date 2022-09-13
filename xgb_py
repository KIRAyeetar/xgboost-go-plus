from sklearn import datasets
import xgboost as xgb
import pandas as pd
import json
import redis
from sklearn import metrics

def save_xgb_feature_map(features, path):
    outfile = open(path, 'w')
    i = 0
    for feat in features:
        outfile.write('{0} {1} q\n'.format(i, feat))
        i = i + 1
    outfile.close()


# 读取鸢尾花数据
iris = datasets.load_iris()
# 特征数据
iris_data = pd.DataFrame(datasets.load_iris().data)
col_names = ["sl", "sw", "pl", "pw"]
iris_data.columns = col_names
# 标签数据
iris_data['target'] = iris.target
# 只取前100条数据，标签只有0 1，并打散
iris_data = iris_data.head(100)
iris_data = iris_data.sample(frac=1, random_state=10)
# 拆分训练集测试集
train = iris_data.head(80)
test = iris_data.tail(20)

# 训练模型
dtrain = xgb.DMatrix(train.loc[:, col_names], label=train["target"])
dtest = xgb.DMatrix(test.loc[:, col_names], label=test["target"])
params = {
    'objective': 'reg:logistic',
    'eta': 0.01,
    'max_depth': 7,
}
bst = xgb.train(params, dtrain, num_boost_round=50)

# 查看预测的结果，可有可无
res = bst.predict(dtest)
print(res)
print(metrics.roc_auc_score(test["target"], res))

# 保存模型文件至本地
bst.dump_model('iris_xgboost_dump.json', dump_format='json')
bst.save_model('iris_xgboost.model')
save_xgb_feature_map(train.loc[:, ["sl", "sw", "pl", "pw"]].columns, "fmap.map")

# 保存模型文件至redis
with open('iris_xgboost_dump.json') as f:
    model_json = json.load(f)
redis_conf = '/data/config/rcmd-mart/rcmd_config.toml'

# 你的json配置
rd = redis.StrictRedis("...")
rd.set("xgb_model_test", str(json.dumps(model_json)), ex=3600)
# 这里是直接传了树深度，Go那里就不需要设置了, 但是key要对应
rd.set("xgb_map_test", str(json.dumps({"sl": 0, "sw": 1, "pl": 2, "pw": 3, "_maxDepth_": 7})), ex=3600)
