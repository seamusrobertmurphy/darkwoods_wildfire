Remote sensing of wildfire burn severity; Darkwoods Conservation Area
================
SMurphy
2020-08-29

- [1. Model Designs](#1-model-designs)
  - [1.1 Import Data](#11-import-data)
  - [1.2 Model Slope](#12-model-slope)
  - [1.3 Model Training](#13-model-training)
  - [1.4 Model Tuning](#14-model-tuning)
  - [1.5 Model Validation](#15-model-validation)
- [2. Full Model List](#2-full-model-list)

## 1. Model Designs

### 1.1 Import Data

Set seed for replication to `123`. Import excel data representing
training data of burn severity classes from candidate spectral indices
sampled in-field in composite burn plots.

``` r
set.seed(123)
darkwoods_fire_plots_data <- read_excel("3.1.darkwoods_fire_ground_plots.xls")
```

### 1.2 Model Slope

``` r
ndbr_lm <- lm(CBI_total_2 ~ NDBR_2016, data = darkwoods_fire_plots_data)
ndbr_lm_resid <- resid(ndbr_lm)
plot(CBI_total_2 ~ NDBR_2016, data = darkwoods_fire_plots_data,
    ylab = "Fire severity score of composite burn plots", xlab = "NDBR",
    col = "blue")
abline(ndbr_lm, col = "red")
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

### 1.3 Model Training

Splitting data 70:30 between training and test subsets according to
distribution of response variable: `CBI_total_2`.

``` r
training.samples <- createDataPartition(darkwoods_fire_plots_data$CBI_total_2,
    p = 0.7, list = FALSE)
train.data <- darkwoods_fire_plots_data[training.samples, ]
test.data <- darkwoods_fire_plots_data[-training.samples, ]
```

Training regimes set by 10K-fold cross validation with 10, 5 and no
repeats using `repeatedcv` and `cv` parameters.

``` r
model_training_10kfold_10repeat <- trainControl(method = "repeatedcv",
    number = 10, repeats = 10, savePredictions = TRUE)


model_training_10kfold_3repeat <- trainControl(method = "repeatedcv",
    number = 10, repeats = 3, savePredictions = TRUE)


model_training_10kfold <- trainControl(method = "cv", number = 10,
    savePredictions = TRUE)
animation::cv.ani(k = 10)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-4-3.png)<!-- -->![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-4-4.png)<!-- -->![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-4-5.png)<!-- -->![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-4-6.png)<!-- -->![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-4-7.png)<!-- -->![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-4-8.png)<!-- -->![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-4-9.png)<!-- -->![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-4-10.png)<!-- -->

### 1.4 Model Tuning

Apply 10K-fold-X10 training regime to test prediction of NDBR between
three candidate models including linear, non-linear support vector
machine regressions, randomForest regression tree using `svmLinear` ,
`svmRadial` , `rf` kernels. Data preprocessing included using `center`
and `scale` operations.

``` r
svm_ndbr_log_linear <- train(CBI_total_2 ~ NDBR_2016, data = train.data,
    method = "svmLinear", trControl = model_training_10kfold,
    preProcess = c("center", "scale"), tuneGrid = expand.grid(C = seq(0,
        3, length = 10)), metric = "RMSE")

svm_ndbr_log_radial <- train(CBI_total_2 ~ NDBR_2016, data = train.data,
    method = "svmRadial", trControl = model_training_10kfold,
    preProcess = c("center", "scale"), tunelength = 10)

rf_nndbr_1000trees = train(CBI_total_2 ~ NDBR_2016, data = train.data,
    method = "rf", ntree = 1000, metric = "RMSE", trControl = model_training_10kfold,
    importance = TRUE)

# Selected kernel
svm_ndbr_log_linear_full <- train(CBI_total_2 ~ NDBR_2016, data = darkwoods_fire_plots_data,
    method = "svmLinear", trControl = model_training_10kfold,
    preProcess = c("center", "scale"), tuneGrid = expand.grid(C = seq(0,
        3, length = 10)), metric = "RMSE")
```

### 1.5 Model Validation

Compare performance of `svmLinear` model between training, test, and
full dataset, using absolute and relative MAE, RMSE, and RMSE-ratio
metrics, and `TheilU` statistic of model bias. Results tables x 2 below
were taken from manuscript section 2.3 - both needing updates.

``` r
svm_ndbr_pred_train <- predict(svm_ndbr_log_linear, data = train.data)
svm_ndbr_pred_test <- predict(svm_ndbr_log_linear, data = test.data)
svm_ndbr_pred_full <- predict(svm_ndbr_log_linear_full, data = darkwoods_fire_plots_data)
svm_ndbr_pred_full_r2 = MLmetrics::R2_Score(svm_ndbr_pred_full,
    darkwoods_fire_plots_data$CBI_total_2)
svm_ndbr_pred_train_r2 = MLmetrics::R2_Score(svm_ndbr_pred_train,
    train.data$CBI_total_2)
svm_ndbr_pred_test_r2 = MLmetrics::R2_Score(svm_ndbr_pred_test,
    test.data$CBI_total_2)
svm_ndbr_pred_full_mae <- mae(svm_ndbr_pred_full, darkwoods_fire_plots_data$CBI_total_2)
svm_ndbr_pred_train_mae <- mae(svm_ndbr_pred_train, train.data$CBI_total_2)
svm_ndbr_pred_test_mae <- mae(svm_ndbr_pred_test, test.data$CBI_total_2)
svm_ndbr_pred_full_mae_rel <- (svm_ndbr_pred_full_mae/mean(darkwoods_fire_plots_data$CBI_total_2)) *
    100
svm_ndbr_pred_train_mae_rel <- (svm_ndbr_pred_train_mae/mean(train.data$CBI_total_2)) *
    100
svm_ndbr_pred_test_mae_rel <- (svm_ndbr_pred_test_mae/mean(test.data$CBI_total_2)) *
    100
svm_ndbr_pred_full_rmse <- rmse(svm_ndbr_pred_full, darkwoods_fire_plots_data$CBI_total_2)
svm_ndbr_pred_train_rmse <- rmse(svm_ndbr_pred_train, train.data$CBI_total_2)
svm_ndbr_pred_test_rmse <- rmse(svm_ndbr_pred_test, test.data$CBI_total_2)
svm_ndbr_pred_full_rmse_rel <- (svm_ndbr_pred_full_rmse/mean(darkwoods_fire_plots_data$CBI_total_2)) *
    100
svm_ndbr_pred_train_rmse_rel <- (svm_ndbr_pred_train_rmse/mean(train.data$CBI_total_2)) *
    100
svm_ndbr_pred_test_rmse_rel <- (svm_ndbr_pred_test_rmse/mean(test.data$CBI_total_2)) *
    100
svm_ndbr_pred_train_rmseRatio = svm_ndbr_pred_train_rmse/svm_ndbr_pred_test_rmse
svm_ndbr_pred_test_rmseRatio = svm_ndbr_pred_test_rmse/svm_ndbr_pred_train_rmse
svm_ndbr_full_Ubias_DescTools = DescTools::TheilU(darkwoods_fire_plots_data$CBI_total_2,
    svm_ndbr_pred_full, type = 2)

svm_ndbr_pred_full_r2
```

    [1] 0.8822122

``` r
svm_ndbr_pred_train_r2
```

    [1] 0.9046967

``` r
svm_ndbr_pred_test_r2
```

    [1] -4.316729

``` r
svm_ndbr_pred_full_mae
```

    [1] 0.6713043

``` r
svm_ndbr_pred_train_mae
```

    [1] 0.5238902

``` r
svm_ndbr_pred_test_mae
```

    [1] 4.022414

``` r
svm_ndbr_pred_full_mae_rel
```

    [1] 15.93547

``` r
svm_ndbr_pred_train_mae_rel
```

    [1] 12.64298

``` r
svm_ndbr_pred_test_mae_rel
```

    [1] 91.56658

``` r
svm_ndbr_pred_full_rmse
```

    [1] 1.008688

``` r
svm_ndbr_pred_train_rmse
```

    [1] 0.8607235

``` r
svm_ndbr_pred_test_rmse
```

    [1] 4.774957

``` r
svm_ndbr_pred_full_rmse_rel
```

    [1] 23.94431

``` r
svm_ndbr_pred_train_rmse_rel
```

    [1] 20.77174

``` r
svm_ndbr_pred_test_rmse_rel
```

    [1] 108.6975

``` r
svm_ndbr_pred_train_rmseRatio
```

    [1] 0.1802578

``` r
svm_ndbr_pred_test_rmseRatio
```

    [1] 5.547609

``` r
svm_ndbr_full_Ubias_DescTools
```

    [1] 0.1963738

``` r
svm_ndbr_log_linear_full$finalModel
```

    Support Vector Machine object of class "ksvm" 

    SV type: eps-svr  (regression) 
     parameter : epsilon = 0.1  cost C = 0.333333333333333 

    Linear (vanilla) kernel function. 

    Number of Support Vectors : 57 

    Objective Function Value : -5.0484 
    Training error : 0.116535 

|           |             |          |                          |            |       |       |
|:---------:|:-----------:|:--------:|:------------------------:|:----------:|:-----:|:-----:|
| **Model** |   **R2**    | **RMSE** | **RMSE<sup>ratio</sup>** | **TheilU** | **C** | **ε** |
|    NBR    | 0.879\*\*\* |   1.05   |           0.46           |   0.035    | 0.336 | 0.10  |
|  BAI2-SL  | 0.183\*\*\* |   2.61   |           1.09           |   0.307    | 1.105 | 0.10  |
|   MIRBI   | 0.551\*\*\* |   1.94   |           1.16           |   0.469    | 0.316 | 0.10  |
|    TVI    | 0.584\*\*\* |   1.88   |           1.16           |   0.432    | 0.474 | 0.10  |
|  TVI-SW   | 0.241\*\*\* |   2.54   |           1.10           |   0.343    | 0.158 | 0.10  |

## 2. Full Model List

``` r
# model 2 - NDBR-SW
svm_ndbr_sw_linear <- train(CBI_total_2 ~ NDBR_SW2_2, data = train.data,
    method = "svmLinear", trControl = model_training_10kfold,
    preProcess = c("center", "scale"), tuneGrid = expand.grid(C = seq(0,
        3, length = 20)))

# model 3 - BAI2
svm_bai2_linear <- train(CBI_total_2 ~ BAI2_2016, data = train.data,
    method = "svmLinear", trControl = model_training_10kfold,
    preProcess = c("center", "scale"), tuneGrid = expand.grid(C = seq(0,
        3, length = 20)))

# model 4 - MIRBI
svm_mirbi_linear <- train(CBI_total_2 ~ MIRBI_2016, data = train.data,
    method = "svmLinear", trControl = model_training_10kfold,
    preProcess = c("center", "scale"), tuneGrid = expand.grid(C = seq(0,
        3, length = 20)))

# model 5 - MSAV
svm_msav_linear <- train(CBI_total_2 ~ MSAV_2016, data = train.data,
    method = "svmLinear", trControl = model_training_10kfold,
    preProcess = c("center", "scale"), tuneGrid = expand.grid(C = seq(0,
        3, length = 20)))

# model 6 - TVI
svm_tvi_linear <- train(CBI_total_2 ~ VIT2_2016, data = train.data,
    method = "svmLinear", trControl = model_training_10kfold,
    preProcess = c("center", "scale"), tuneGrid = expand.grid(C = seq(0,
        3, length = 20)))

# model 7 - TVI-SW
svm_tvi_sw_linear <- train(CBI_total_2 ~ VIRSW2_201, data = train.data,
    method = "svmLinear", trControl = model_training_10kfold,
    preProcess = c("center", "scale"), tuneGrid = expand.grid(C = seq(0,
        3, length = 20)))
```

``` r
# model 2 - NDBRSW results
summary(lm(predict(svm_ndbr_sw_linear) ~ train.data$CBI_total_2))
```


    Call:
    lm(formula = predict(svm_ndbr_sw_linear) ~ train.data$CBI_total_2)

    Residuals:
        Min      1Q  Median      3Q     Max 
    -2.1946 -0.7337 -0.1164  0.6918  2.1474 

    Coefficients:
                           Estimate Std. Error t value Pr(>|t|)    
    (Intercept)             3.40877    0.21770  15.658   <2e-16 ***
    train.data$CBI_total_2  0.08340    0.04359   1.913   0.0601 .  
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Residual standard error: 1.002 on 66 degrees of freedom
    Multiple R-squared:  0.05255,   Adjusted R-squared:  0.03819 
    F-statistic: 3.661 on 1 and 66 DF,  p-value: 0.06005

``` r
svm_ndbr_sw_pred_train <- predict(svm_ndbr_sw_linear, data = train.data)
svm_ndbr_sw_pred_train_mae <- mae(svm_ndbr_sw_pred_train, train.data$CBI_total_2)
svm_ndbr_sw_pred_train_mae
```

    [1] 2.294951

``` r
svm_ndbr_sw_pred_train_mae_rel <- (svm_ndbr_sw_pred_train_mae/mean(train.data$CBI_total_2)) *
    100
svm_ndbr_sw_pred_train_mae_rel
```

    [1] 55.38379

``` r
svm_ndbr_sw_pred_train_rmse <- rmse(svm_ndbr_sw_pred_train, train.data$CBI_total_2)
svm_ndbr_sw_pred_train_rmse
```

    [1] 2.767207

``` r
svm_ndbr_sw_pred_train_rmse_rel <- (svm_ndbr_sw_pred_train_rmse/mean(train.data$CBI_total_2)) *
    100
svm_ndbr_sw_pred_train_rmse_rel
```

    [1] 66.78068

``` r
svm_ndbr_sw_pred_train_R2 <- R2(svm_ndbr_sw_pred_train, train.data$CBI_total_2)
svm_ndbr_sw_pred_train_R2
```

    [1] 0.05254919

``` r
TheilU(train.data$CBI_total_2, svm_ndbr_sw_pred_train, type = 2)
```

    [1] 0.5540623

``` r
svm_ndbr_sw_train_Ubias = DescTools::TheilU(train.data$CBI_total_2,
    svm_ndbr_sw_pred_train, type = 2)

svm_ndbr_sw_train_Ubias <- ((svm_ndbr_sw_pred_train_mae) * 68)/((svm_ndbr_sw_pred_train_mae)^2)
svm_ndbr_sw_train_Ubias
```

    [1] 29.63026

``` r
svm_ndbr_sw_pred_test <- predict(svm_ndbr_sw_linear, data = darkwoods_fire_plots_data)
svm_ndbr_sw_pred_test_rmse <- rmse(svm_ndbr_sw_pred_test, darkwoods_fire_plots_data$CBI_total_2)
svm_ndbr_sw_pred_test_rmse
```

    [1] 2.913245

``` r
svm_ndbr_sw_pred_test_rmse_rel <- (svm_ndbr_sw_pred_test_rmse/mean(darkwoods_fire_plots_data$CBI_total_2)) *
    100
svm_ndbr_sw_pred_test_rmse_rel
```

    [1] 69.15485

``` r
svm_ndbr_sw_pred_train_rmse/svm_ndbr_sw_pred_test_rmse
```

    [1] 0.9498709

``` r
svm_ndbr_sw_pred_test_rmse/svm_ndbr_sw_pred_train_rmse
```

    [1] 1.052775

``` r
svm_ndbr_sw_linear
```

    Support Vector Machines with Linear Kernel 

    68 samples
     1 predictor

    Pre-processing: centered (1), scaled (1) 
    Resampling: Cross-Validated (10 fold) 
    Summary of sample sizes: 60, 60, 61, 61, 63, 60, ... 
    Resampling results across tuning parameters:

      C          RMSE      Rsquared   MAE     
      0.0000000       NaN        NaN       NaN
      0.1578947  2.815026  0.1192282  2.420296
      0.3157895  2.838784  0.1192282  2.434799
      0.4736842  2.836611  0.1192282  2.431870
      0.6315789  2.838615  0.1192282  2.433516
      0.7894737  2.838578  0.1192282  2.433531
      0.9473684  2.838592  0.1192282  2.433502
      1.1052632  2.840875  0.1192282  2.430698
      1.2631579  2.843820  0.1192282  2.426868
      1.4210526  2.843849  0.1192282  2.426874
      1.5789474  2.843855  0.1192282  2.426856
      1.7368421  2.843856  0.1192282  2.426881
      1.8947368  2.843820  0.1192282  2.426836
      2.0526316  2.843855  0.1192282  2.426856
      2.2105263  2.843856  0.1192282  2.426874
      2.3684211  2.843815  0.1192282  2.426834
      2.5263158  2.843855  0.1192282  2.426856
      2.6842105  2.843856  0.1192282  2.426867
      2.8421053  2.843855  0.1192282  2.426856
      3.0000000  2.843842  0.1192282  2.426848

    RMSE was used to select the optimal model using the smallest value.
    The final value used for the model was C = 0.1578947.

``` r
plot(svm_ndbr_sw_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

``` r
densityplot(svm_ndbr_sw_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-2.png)<!-- -->

``` r
svm_ndbr_sw_linear$finalModel
```

    Support Vector Machine object of class "ksvm" 

    SV type: eps-svr  (regression) 
     parameter : epsilon = 0.1  cost C = 0.157894736842105 

    Linear (vanilla) kernel function. 

    Number of Support Vectors : 66 

    Objective Function Value : -7.77 
    Training error : 0.970576 

``` r
# model 3 - BAI2 results
summary(lm(predict(svm_bai2_linear) ~ train.data$CBI_total_2))
```


    Call:
    lm(formula = predict(svm_bai2_linear) ~ train.data$CBI_total_2)

    Residuals:
        Min      1Q  Median      3Q     Max 
    -3.9223 -0.2897  0.2416  0.9653  1.9853 

    Coefficients:
                           Estimate Std. Error t value Pr(>|t|)    
    (Intercept)              3.4820     0.2927  11.897  < 2e-16 ***
    train.data$CBI_total_2   0.2450     0.0586   4.181 8.72e-05 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Residual standard error: 1.347 on 66 degrees of freedom
    Multiple R-squared:  0.2094,    Adjusted R-squared:  0.1975 
    F-statistic: 17.48 on 1 and 66 DF,  p-value: 8.719e-05

``` r
svm_bai2_pred_train <- predict(svm_bai2_linear, data = train.data)
svm_bai2_pred_train_mae <- mae(svm_bai2_pred_train, train.data$CBI_total_2)
svm_bai2_pred_train_mae
```

    [1] 2.079445

``` r
svm_bai2_pred_train_mae_rel <- (svm_bai2_pred_train_mae/mean(train.data$CBI_total_2)) *
    100
svm_bai2_pred_train_mae_rel
```

    [1] 50.18299

``` r
svm_bai2_pred_train_rmse <- rmse(svm_bai2_pred_train, train.data$CBI_total_2)
svm_bai2_pred_train_rmse
```

    [1] 2.513496

``` r
svm_bai2_pred_train_rmse_rel <- (svm_bai2_pred_train_rmse/mean(train.data$CBI_total_2)) *
    100
svm_bai2_pred_train_rmse_rel
```

    [1] 60.65791

``` r
svm_bai2_pred_train_R2 <- R2(svm_bai2_pred_train, train.data$CBI_total_2)
svm_bai2_pred_train_R2
```

    [1] 0.2094349

``` r
TheilU(train.data$CBI_total_2, svm_bai2_pred_train, type = 2)
```

    [1] 0.5032632

``` r
svm_bai2_train_Ubias <- ((svm_bai2_pred_train_mae) * 68)/((svm_bai2_pred_train_mae)^2)
svm_bai2_train_Ubias
```

    [1] 32.70104

``` r
svm_bai2_pred_test <- predict(svm_bai2_linear, data = darkwoods_fire_plots_data)
svm_bai2_pred_test_rmse <- rmse(svm_bai2_pred_test, darkwoods_fire_plots_data$CBI_total_2)
svm_bai2_pred_test_rmse
```

    [1] 3.297187

``` r
svm_bai2_pred_test_rmse_rel <- (svm_bai2_pred_test_rmse/mean(darkwoods_fire_plots_data$CBI_total_2)) *
    100
svm_bai2_pred_test_rmse_rel
```

    [1] 78.26888

``` r
svm_bai2_pred_train_rmse/svm_bai2_pred_test_rmse
```

    [1] 0.7623154

``` r
svm_bai2_pred_test_rmse/svm_bai2_pred_train_rmse
```

    [1] 1.311793

``` r
svm_bai2_linear
```

    Support Vector Machines with Linear Kernel 

    68 samples
     1 predictor

    Pre-processing: centered (1), scaled (1) 
    Resampling: Cross-Validated (10 fold) 
    Summary of sample sizes: 60, 61, 61, 60, 62, 60, ... 
    Resampling results across tuning parameters:

      C          RMSE      Rsquared  MAE     
      0.0000000       NaN       NaN       NaN
      0.1578947  2.518558  0.300206  2.151726
      0.3157895  2.516652  0.300206  2.146948
      0.4736842  2.516891  0.300206  2.147299
      0.6315789  2.511234  0.300206  2.140987
      0.7894737  2.511589  0.300206  2.136948
      0.9473684  2.511591  0.300206  2.136981
      1.1052632  2.511593  0.300206  2.136948
      1.2631579  2.511595  0.300206  2.136953
      1.4210526  2.510861  0.300206  2.136241
      1.5789474  2.510962  0.300206  2.136144
      1.7368421  2.510954  0.300206  2.136135
      1.8947368  2.510950  0.300206  2.136131
      2.0526316  2.510962  0.300206  2.136144
      2.2105263  2.510961  0.300206  2.136143
      2.3684211  2.510895  0.300206  2.136084
      2.5263158  2.510961  0.300206  2.136143
      2.6842105  2.515528  0.300206  2.141175
      2.8421053  2.524126  0.300206  2.150439
      3.0000000  2.525354  0.300206  2.151864

    RMSE was used to select the optimal model using the smallest value.
    The final value used for the model was C = 1.421053.

``` r
plot(svm_bai2_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-3.png)<!-- -->

``` r
densityplot(svm_bai2_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-4.png)<!-- -->

``` r
svm_bai2_linear$finalModel
```

    Support Vector Machine object of class "ksvm" 

    SV type: eps-svr  (regression) 
     parameter : epsilon = 0.1  cost C = 1.42105263157895 

    Linear (vanilla) kernel function. 

    Number of Support Vectors : 65 

    Objective Function Value : -62.1944 
    Training error : 0.800761 

``` r
plot(svm_bai2_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-5.png)<!-- -->

``` r
# model 4 - MIRBI results
summary(lm(predict(svm_mirbi_linear) ~ train.data$CBI_total_2))
```


    Call:
    lm(formula = predict(svm_mirbi_linear) ~ train.data$CBI_total_2)

    Residuals:
        Min      1Q  Median      3Q     Max 
    -4.2400 -0.8581  0.3351  0.9569  2.8909 

    Coefficients:
                           Estimate Std. Error t value Pr(>|t|)    
    (Intercept)             1.93909    0.30559   6.345 2.32e-08 ***
    train.data$CBI_total_2  0.54293    0.06119   8.873 7.30e-13 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Residual standard error: 1.407 on 66 degrees of freedom
    Multiple R-squared:  0.544, Adjusted R-squared:  0.5371 
    F-statistic: 78.74 on 1 and 66 DF,  p-value: 7.298e-13

``` r
svm_mirbi_pred_train <- predict(svm_mirbi_linear, data = train.data)
svm_mirbi_pred_train_mae <- mae(svm_mirbi_pred_train, train.data$CBI_total_2)
svm_mirbi_pred_train_mae
```

    [1] 1.387107

``` r
svm_mirbi_pred_train_mae_rel <- (svm_mirbi_pred_train_mae/mean(train.data$CBI_total_2)) *
    100
svm_mirbi_pred_train_mae_rel
```

    [1] 33.47489

``` r
svm_mirbi_pred_train_rmse <- rmse(svm_mirbi_pred_train, train.data$CBI_total_2)
svm_mirbi_pred_train_rmse
```

    [1] 1.883282

``` r
svm_mirbi_pred_train_rmse_rel <- (svm_mirbi_pred_train_rmse/mean(train.data$CBI_total_2)) *
    100
svm_mirbi_pred_train_rmse_rel
```

    [1] 45.44903

``` r
svm_mirbi_pred_train_R2 <- R2(svm_mirbi_pred_train, train.data$CBI_total_2)
svm_mirbi_pred_train_R2
```

    [1] 0.5440053

``` r
TheilU(train.data$CBI_total_2, svm_mirbi_pred_train, type = 2)
```

    [1] 0.377079

``` r
svm_mirbi_train_Ubias <- ((svm_mirbi_pred_train_mae) * 68)/((svm_mirbi_pred_train_mae)^2)
svm_mirbi_train_Ubias
```

    [1] 49.0229

``` r
svm_mirbi_pred_test <- predict(svm_mirbi_linear, data = darkwoods_fire_plots_data)
svm_mirbi_pred_test_rmse <- rmse(svm_mirbi_pred_test, darkwoods_fire_plots_data$CBI_total_2)
svm_mirbi_pred_test_rmse
```

    [1] 3.421362

``` r
svm_mirbi_pred_test_rmse_rel <- (svm_mirbi_pred_test_rmse/mean(darkwoods_fire_plots_data$CBI_total_2)) *
    100
svm_mirbi_pred_test_rmse_rel
```

    [1] 81.21656

``` r
svm_mirbi_pred_train_rmse/svm_mirbi_pred_test_rmse
```

    [1] 0.5504482

``` r
svm_mirbi_pred_test_rmse/svm_mirbi_pred_train_rmse
```

    [1] 1.816701

``` r
svm_mirbi_linear
```

    Support Vector Machines with Linear Kernel 

    68 samples
     1 predictor

    Pre-processing: centered (1), scaled (1) 
    Resampling: Cross-Validated (10 fold) 
    Summary of sample sizes: 63, 62, 60, 62, 61, 60, ... 
    Resampling results across tuning parameters:

      C          RMSE      Rsquared   MAE     
      0.0000000       NaN        NaN       NaN
      0.1578947  1.798969  0.6119423  1.410038
      0.3157895  1.792493  0.6119423  1.408514
      0.4736842  1.795557  0.6119423  1.408886
      0.6315789  1.794888  0.6119423  1.409210
      0.7894737  1.794926  0.6119423  1.409204
      0.9473684  1.794888  0.6119423  1.409210
      1.1052632  1.794892  0.6119423  1.409209
      1.2631579  1.794888  0.6119423  1.409210
      1.4210526  1.794719  0.6119423  1.407268
      1.5789474  1.793983  0.6119423  1.406145
      1.7368421  1.793931  0.6119423  1.406152
      1.8947368  1.793927  0.6119423  1.406160
      2.0526316  1.793539  0.6119423  1.406067
      2.2105263  1.793586  0.6119423  1.406083
      2.3684211  1.793547  0.6119423  1.406073
      2.5263158  1.793609  0.6119423  1.406096
      2.6842105  1.793606  0.6119423  1.406077
      2.8421053  1.793539  0.6119423  1.406067
      3.0000000  1.793539  0.6119423  1.406067

    RMSE was used to select the optimal model using the smallest value.
    The final value used for the model was C = 0.3157895.

``` r
plot(svm_mirbi_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-6.png)<!-- -->

``` r
densityplot(svm_mirbi_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-7.png)<!-- -->

``` r
svm_mirbi_linear$finalModel
```

    Support Vector Machine object of class "ksvm" 

    SV type: eps-svr  (regression) 
     parameter : epsilon = 0.1  cost C = 0.315789473684211 

    Linear (vanilla) kernel function. 

    Number of Support Vectors : 62 

    Objective Function Value : -8.8373 
    Training error : 0.449549 

``` r
plot(svm_mirbi_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-8.png)<!-- -->

``` r
# model 5 - MSAV results
summary(lm(predict(svm_msav_linear) ~ train.data$CBI_total_2))
```


    Call:
    lm(formula = predict(svm_msav_linear) ~ train.data$CBI_total_2)

    Residuals:
        Min      1Q  Median      3Q     Max 
    -1.6722 -0.9243 -0.0853  0.6957  3.3895 

    Coefficients:
                           Estimate Std. Error t value Pr(>|t|)    
    (Intercept)              3.6308     0.2362  15.370  < 2e-16 ***
    train.data$CBI_total_2   0.1739     0.0473   3.676 0.000477 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Residual standard error: 1.087 on 66 degrees of freedom
    Multiple R-squared:   0.17, Adjusted R-squared:  0.1574 
    F-statistic: 13.51 on 1 and 66 DF,  p-value: 0.000477

``` r
svm_msav_pred_train <- predict(svm_msav_linear, data = train.data)
svm_msav_pred_train_mae <- mae(svm_msav_pred_train, train.data$CBI_total_2)
svm_msav_pred_train_mae
```

    [1] 2.24269

``` r
svm_msav_pred_train_mae_rel <- (svm_msav_pred_train_mae/mean(train.data$CBI_total_2)) *
    100
svm_msav_pred_train_mae_rel
```

    [1] 54.12257

``` r
svm_msav_pred_train_rmse <- rmse(svm_msav_pred_train, train.data$CBI_total_2)
svm_msav_pred_train_rmse
```

    [1] 2.548744

``` r
svm_msav_pred_train_rmse_rel <- (svm_msav_pred_train_rmse/mean(train.data$CBI_total_2)) *
    100
svm_msav_pred_train_rmse_rel
```

    [1] 61.50855

``` r
svm_msav_pred_train_R2 <- R2(svm_msav_pred_train, train.data$CBI_total_2)
svm_msav_pred_train_R2
```

    [1] 0.1699657

``` r
TheilU(train.data$CBI_total_2, svm_msav_pred_train, type = 2)
```

    [1] 0.5103207

``` r
svm_msav_train_Ubias <- ((svm_msav_pred_train_mae) * 68)/((svm_msav_pred_train_mae)^2)
svm_msav_train_Ubias
```

    [1] 30.32073

``` r
svm_msav_pred_test <- predict(svm_msav_linear, data = darkwoods_fire_plots_data)
svm_msav_pred_test_rmse <- rmse(svm_msav_pred_test, darkwoods_fire_plots_data$CBI_total_2)
svm_msav_pred_test_rmse
```

    [1] 3.091682

``` r
svm_msav_pred_test_rmse_rel <- (svm_msav_pred_test_rmse/mean(darkwoods_fire_plots_data$CBI_total_2)) *
    100
svm_msav_pred_test_rmse_rel
```

    [1] 73.3906

``` r
svm_msav_pred_train_rmse/svm_msav_pred_test_rmse
```

    [1] 0.8243875

``` r
svm_msav_pred_test_rmse/svm_msav_pred_train_rmse
```

    [1] 1.213022

``` r
svm_msav_linear
```

    Support Vector Machines with Linear Kernel 

    68 samples
     1 predictor

    Pre-processing: centered (1), scaled (1) 
    Resampling: Cross-Validated (10 fold) 
    Summary of sample sizes: 63, 60, 61, 61, 63, 61, ... 
    Resampling results across tuning parameters:

      C          RMSE      Rsquared   MAE     
      0.0000000       NaN        NaN       NaN
      0.1578947  2.568677  0.2151577  2.302789
      0.3157895  2.575231  0.2151577  2.309187
      0.4736842  2.574839  0.2151577  2.308759
      0.6315789  2.575128  0.2151577  2.309151
      0.7894737  2.575128  0.2151577  2.309151
      0.9473684  2.575128  0.2151577  2.309151
      1.1052632  2.575101  0.2151577  2.309105
      1.2631579  2.575128  0.2151577  2.309151
      1.4210526  2.575128  0.2151577  2.309151
      1.5789474  2.575128  0.2151577  2.309151
      1.7368421  2.575128  0.2151577  2.309151
      1.8947368  2.575128  0.2151577  2.309151
      2.0526316  2.575128  0.2151577  2.309151
      2.2105263  2.575128  0.2151577  2.309151
      2.3684211  2.575128  0.2151577  2.309151
      2.5263158  2.575128  0.2151577  2.309151
      2.6842105  2.575128  0.2151577  2.309151
      2.8421053  2.575128  0.2151577  2.309151
      3.0000000  2.573090  0.2151577  2.307642

    RMSE was used to select the optimal model using the smallest value.
    The final value used for the model was C = 0.1578947.

``` r
plot(svm_msav_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-9.png)<!-- -->

``` r
densityplot(svm_msav_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-10.png)<!-- -->

``` r
svm_msav_linear$finalModel
```

    Support Vector Machine object of class "ksvm" 

    SV type: eps-svr  (regression) 
     parameter : epsilon = 0.1  cost C = 0.157894736842105 

    Linear (vanilla) kernel function. 

    Number of Support Vectors : 65 

    Objective Function Value : -7.6244 
    Training error : 0.823377 

``` r
plot(svm_msav_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-11.png)<!-- -->

``` r
# model 6 - TVI results
summary(lm(predict(svm_tvi_linear) ~ train.data$CBI_total_2))
```


    Call:
    lm(formula = predict(svm_tvi_linear) ~ train.data$CBI_total_2)

    Residuals:
        Min      1Q  Median      3Q     Max 
    -2.2991 -0.8968 -0.1297  0.7738  4.0474 

    Coefficients:
                           Estimate Std. Error t value Pr(>|t|)    
    (Intercept)             2.23292    0.28653   7.793 6.22e-11 ***
    train.data$CBI_total_2  0.50138    0.05737   8.739 1.26e-12 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Residual standard error: 1.319 on 66 degrees of freedom
    Multiple R-squared:  0.5364,    Adjusted R-squared:  0.5294 
    F-statistic: 76.38 on 1 and 66 DF,  p-value: 1.264e-12

``` r
svm_tvi_pred_train <- predict(svm_tvi_linear, data = train.data)
svm_tvi_pred_train_mae <- mae(svm_tvi_pred_train, train.data$CBI_total_2)
svm_tvi_pred_train_mae
```

    [1] 1.64125

``` r
svm_tvi_pred_train_mae_rel <- (svm_tvi_pred_train_mae/mean(train.data$CBI_total_2)) *
    100
svm_tvi_pred_train_mae_rel
```

    [1] 39.60809

``` r
svm_tvi_pred_train_rmse <- rmse(svm_tvi_pred_train, train.data$CBI_total_2)
svm_tvi_pred_train_rmse
```

    [1] 1.910272

``` r
svm_tvi_pred_train_rmse_rel <- (svm_tvi_pred_train_rmse/mean(train.data$CBI_total_2)) *
    100
svm_tvi_pred_train_rmse_rel
```

    [1] 46.10036

``` r
svm_tvi_pred_train_R2 <- R2(svm_tvi_pred_train, train.data$CBI_total_2)
svm_tvi_pred_train_R2
```

    [1] 0.5364397

``` r
TheilU(train.data$CBI_total_2, svm_tvi_pred_train, type = 2)
```

    [1] 0.3824829

``` r
svm_tvi_train_Ubias <- ((svm_tvi_pred_train_mae) * 68)/((svm_tvi_pred_train_mae)^2)
svm_tvi_train_Ubias
```

    [1] 41.43184

``` r
svm_tvi_pred_test <- predict(svm_tvi_linear, data = darkwoods_fire_plots_data)
svm_tvi_pred_test_rmse <- rmse(svm_tvi_pred_test, darkwoods_fire_plots_data$CBI_total_2)
svm_tvi_pred_test_rmse
```

    [1] 3.439511

``` r
svm_tvi_pred_test_rmse_rel <- (svm_tvi_pred_test_rmse/mean(darkwoods_fire_plots_data$CBI_total_2)) *
    100
svm_tvi_pred_test_rmse_rel
```

    [1] 81.64739

``` r
svm_tvi_pred_train_rmse/svm_tvi_pred_test_rmse
```

    [1] 0.5553904

``` r
svm_tvi_pred_test_rmse/svm_tvi_pred_train_rmse
```

    [1] 1.800535

``` r
svm_tvi_linear
```

    Support Vector Machines with Linear Kernel 

    68 samples
     1 predictor

    Pre-processing: centered (1), scaled (1) 
    Resampling: Cross-Validated (10 fold) 
    Summary of sample sizes: 60, 61, 62, 60, 61, 62, ... 
    Resampling results across tuning parameters:

      C          RMSE      Rsquared   MAE     
      0.0000000       NaN        NaN       NaN
      0.1578947  1.895250  0.6337241  1.683869
      0.3157895  1.902632  0.6337241  1.687800
      0.4736842  1.902352  0.6337241  1.688175
      0.6315789  1.898315  0.6337241  1.681712
      0.7894737  1.897790  0.6337241  1.681141
      0.9473684  1.898016  0.6337241  1.681808
      1.1052632  1.898014  0.6337241  1.681806
      1.2631579  1.898014  0.6337241  1.681806
      1.4210526  1.897901  0.6337241  1.681427
      1.5789474  1.897109  0.6337241  1.678989
      1.7368421  1.896900  0.6337241  1.678757
      1.8947368  1.896921  0.6337241  1.678779
      2.0526316  1.896922  0.6337241  1.678777
      2.2105263  1.905584  0.6337241  1.690103
      2.3684211  1.903041  0.6337241  1.687142
      2.5263158  1.902964  0.6337241  1.687060
      2.6842105  1.902964  0.6337241  1.687060
      2.8421053  1.902917  0.6337241  1.687008
      3.0000000  1.902924  0.6337241  1.687015

    RMSE was used to select the optimal model using the smallest value.
    The final value used for the model was C = 0.1578947.

``` r
plot(svm_tvi_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-12.png)<!-- -->

``` r
densityplot(svm_tvi_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-13.png)<!-- -->

``` r
svm_tvi_linear$finalModel
```

    Support Vector Machine object of class "ksvm" 

    SV type: eps-svr  (regression) 
     parameter : epsilon = 0.1  cost C = 0.157894736842105 

    Linear (vanilla) kernel function. 

    Number of Support Vectors : 62 

    Objective Function Value : -5.4886 
    Training error : 0.462526 

``` r
plot(svm_tvi_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-14.png)<!-- -->

``` r
# model 7 - TVI-SW results
summary(lm(predict(svm_tvi_sw_linear) ~ train.data$CBI_total_2))
```


    Call:
    lm(formula = predict(svm_tvi_sw_linear) ~ train.data$CBI_total_2)

    Residuals:
        Min      1Q  Median      3Q     Max 
    -4.0366 -1.1358  0.1227  1.1706  2.4976 

    Coefficients:
                           Estimate Std. Error t value Pr(>|t|)    
    (Intercept)             2.68832    0.32154   8.361 5.99e-12 ***
    train.data$CBI_total_2  0.34301    0.06438   5.328 1.29e-06 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Residual standard error: 1.48 on 66 degrees of freedom
    Multiple R-squared:  0.3008,    Adjusted R-squared:  0.2902 
    F-statistic: 28.39 on 1 and 66 DF,  p-value: 1.29e-06

``` r
svm_tvi_sw_pred_train <- predict(svm_tvi_sw_linear, data = train.data)
svm_tvi_sw_pred_train_mae <- mae(svm_tvi_sw_pred_train, train.data$CBI_total_2)
svm_tvi_sw_pred_train_mae
```

    [1] 1.805267

``` r
svm_tvi_sw_pred_train_mae_rel <- (svm_tvi_sw_pred_train_mae/mean(train.data$CBI_total_2)) *
    100
svm_tvi_sw_pred_train_mae_rel
```

    [1] 43.56629

``` r
svm_tvi_sw_pred_train_rmse <- rmse(svm_tvi_sw_pred_train, train.data$CBI_total_2)
svm_tvi_sw_pred_train_rmse
```

    [1] 2.341571

``` r
svm_tvi_sw_pred_train_rmse_rel <- (svm_tvi_sw_pred_train_rmse/mean(train.data$CBI_total_2)) *
    100
svm_tvi_sw_pred_train_rmse_rel
```

    [1] 56.50886

``` r
svm_tvi_sw_pred_train_R2 <- R2(svm_tvi_sw_pred_train, train.data$CBI_total_2)
svm_tvi_sw_pred_train_R2
```

    [1] 0.3007533

``` r
TheilU(train.data$CBI_total_2, svm_tvi_sw_pred_train, type = 2)
```

    [1] 0.4688396

``` r
svm_tvi_sw_train_Ubias <- ((svm_tvi_sw_pred_train_mae) * 68)/((svm_tvi_sw_pred_train_mae)^2)
svm_tvi_sw_train_Ubias
```

    [1] 37.66756

``` r
svm_tvi_sw_pred_test <- predict(svm_tvi_sw_linear, data = darkwoods_fire_plots_data)
svm_tvi_sw_pred_test_rmse <- rmse(svm_tvi_sw_pred_test, darkwoods_fire_plots_data$CBI_total_2)
svm_tvi_sw_pred_test_rmse
```

    [1] 3.278601

``` r
svm_tvi_sw_pred_test_rmse_rel <- (svm_tvi_sw_pred_test_rmse/mean(darkwoods_fire_plots_data$CBI_total_2)) *
    100
svm_tvi_sw_pred_test_rmse_rel
```

    [1] 77.82768

``` r
svm_tvi_sw_pred_train_rmse/svm_tvi_sw_pred_test_rmse
```

    [1] 0.7141983

``` r
svm_tvi_sw_pred_test_rmse/svm_tvi_sw_pred_train_rmse
```

    [1] 1.400171

``` r
svm_tvi_sw_linear
```

    Support Vector Machines with Linear Kernel 

    68 samples
     1 predictor

    Pre-processing: centered (1), scaled (1) 
    Resampling: Cross-Validated (10 fold) 
    Summary of sample sizes: 61, 63, 61, 61, 61, 61, ... 
    Resampling results across tuning parameters:

      C          RMSE      Rsquared   MAE     
      0.0000000       NaN        NaN       NaN
      0.1578947  2.282925  0.4193623  1.867305
      0.3157895  2.270585  0.4193623  1.859384
      0.4736842  2.283809  0.4193623  1.874426
      0.6315789  2.283813  0.4193623  1.874484
      0.7894737  2.283861  0.4193623  1.874546
      0.9473684  2.294611  0.4193623  1.885666
      1.1052632  2.282251  0.4193623  1.870958
      1.2631579  2.281699  0.4193623  1.870129
      1.4210526  2.281699  0.4193623  1.870129
      1.5789474  2.282895  0.4193623  1.870694
      1.7368421  2.287690  0.4193623  1.875080
      1.8947368  2.287693  0.4193623  1.875108
      2.0526316  2.287713  0.4193623  1.875111
      2.2105263  2.287690  0.4193623  1.875080
      2.3684211  2.287690  0.4193623  1.875080
      2.5263158  2.287690  0.4193623  1.875080
      2.6842105  2.287690  0.4193623  1.875079
      2.8421053  2.287706  0.4193623  1.875115
      3.0000000  2.287689  0.4193623  1.875079

    RMSE was used to select the optimal model using the smallest value.
    The final value used for the model was C = 0.3157895.

``` r
plot(svm_tvi_sw_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-15.png)<!-- -->

``` r
densityplot(svm_tvi_sw_linear)
```

![](darkwoods_wildfire_files/figure-gfm/unnamed-chunk-8-16.png)<!-- -->

``` r
svm_tvi_sw_linear$finalModel
```

    Support Vector Machine object of class "ksvm" 

    SV type: eps-svr  (regression) 
     parameter : epsilon = 0.1  cost C = 0.315789473684211 

    Linear (vanilla) kernel function. 

    Number of Support Vectors : 63 

    Objective Function Value : -11.9065 
    Training error : 0.694962 
