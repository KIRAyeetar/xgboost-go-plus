package main

import (
	"awesomeProject/models"
	"fmt"
	"github.com/Elvenson/xgboost-go/mat"
	"github.com/gomodule/redigo/redis"
	"time"
	"github.com/Elvenson/xgboost-go/activation"
	"os"
)

func main() {
	// 构造预测数据, 基于鸢尾花数据
	input := mat.SparseMatrix{
		Vectors: []mat.SparseVector{
			map[int]float64{0: 5.1, 1: 3.3, 2: 1.7, 4: 0.5},
			map[int]float64{0: 5.7, 1: 3.0, 2: 4.2, 4: 1.2},
		},
	}

	// 通过本地文件获取预测结果
	// 树的深度要手动设置下. numClass 和 activation 参照 xgboost-go 设置
	ensembleLoc, err := models.LoadXGBoostFromLocalJSON("iris_xgboost_dump.json", "fmap.map", 1, 7, &activation.Logistic{})
	if err != nil {
		fmt.Println(err)
		os.Exit(0)
	}
	t := time.Now()
	res, err := ensembleLoc.PredictProba(input)
	fmt.Println(res.Vectors[0], err)
	fmt.Println(res.Vectors[1], err)
	fmt.Println(time.Since(t))

	// 通过redis key获取预测结果
	// red 就是使用的redis实力，这里用的redigo, 可以灵活修改
	var red *redis.Pool
	// 树深度已在 python 侧存进了fmap, 所以这里不需要再设置; 第二个返回的结果是从redis里读取的fmap, 看需求保留
	ensembleRed, _, err := models.LoadXGBoostByRedis("xgb_model_test", "xgb_map_test", 1, &activation.Logistic{}, red)
	if err != nil {
		fmt.Println(err)
		os.Exit(0)
	}
	fmt.Println(ensembleRed, err)
	t = time.Now()
	res, err = ensembleRed.PredictProba(input)
	fmt.Println(res.Vectors[0], err)
	fmt.Println(res.Vectors[1], err)
	fmt.Println(time.Since(t))
	os.Exit(0)
}
