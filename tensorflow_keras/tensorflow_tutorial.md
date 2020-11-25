tensorflow tutorials
================
Miguel A. García-Campos

# Description

This notebook presents machine-learning examples using tensorflow/keras
as infrastructure to create and train neural networks.

# Setup: Packages and functions

Loading local functions and external packages

``` r
tStart <- Sys.time()
source("http://bit.ly/rnaMods") # Load my functions
# source("local_funs.R")
CRAN_packs <- c("parallel", "keras", "tidyverse")
invisible(sapply(CRAN_packs, installLoad_CRAN))
# BIOC_packs <- c("")
# invisible(sapply(BIOC_packs, installLoad_BioC))
nCores <- 10 # Number of cores used in multi-core functions
```

# Categorical Classification from continuous variables - Iris dataset

This example is published by **leonjessen** in their github repo
[keras\_tensorflow\_on\_iris](https://github.com/leonjessen/keras_tensorflow_on_iris)

-----

**Building a simple neural network using Keras and Tensorflow**

A minimal example for building your first simple artificial neural
network using [Keras and TensorFlow for
R](https://tensorflow.rstudio.com/keras/) - Right, let’s get to it\!

## Data

[The famous Iris flower data
set](https://en.wikipedia.org/wiki/Iris_flower_data_set) contains data
to quantify the morphologic variation of Iris flowers of three related
species. In other words - A total of 150 observations of 4 input
features `Sepal.Length`, `Sepal.Width`, `Petal.Length` and `Petal.Width`
and 3 output classes `setosa` `versicolor` and `virginica`, with 50
observations in each class. The distributions of the feature values
looks like so:

``` r
iris %>% as_tibble %>% gather(feature, value, -Species) %>%
  ggplot(aes(x = feature, y = value, fill = Species)) +
  geom_violin(alpha = 0.5, scale = "width", position = position_dodge(width = 0.9)) +
  geom_boxplot(alpha = 0.5, width = 0.2, position = position_dodge(width = 0.9)) +
  theme_bw() + ggtitle("Iris dataset")
```

![](tensorflow_tutorial_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

## Aim

Our aim is to connect the 4 input features (`Sepal.Length`,
`Sepal.Width`, `Petal.Length` and `Petal.Width`) to the correct output
class (`setosa` `versicolor` and `virginica`) using an artificial neural
network. For this task, we have chosen the following simple architecture
with one input layer with 4 neurons (one for each feature), one hidden
layer with 4 neurons and one output layer with 3 neurons (one for each
class), all fully connected:

Our artificial neural network will have a total of 35 parameters: 4 for
each input neuron connected to the hidden layer, plus an additional 4
for the associated first bias neuron and 3 for each of the hidden
neurons connected to the output layer, plus an additional 3 for the
associated second bias neuron. I.e. (4 x 4)+ 4 + (4 x 3) + 3 = 35

## Prepare data

We start with slightly wrangling the iris data set by renaming and
scaling the features and converting character labels to numeric:

``` r
nn_dat = iris %>% as_tibble %>%
  mutate(sepal_l_feat = scale(Sepal.Length),
         sepal_w_feat = scale(Sepal.Width),
         petal_l_feat = scale(Petal.Length),
         petal_w_feat = scale(Petal.Width),          
         class_num    = as.numeric(Species) - 1, # factor, so = 0, 1, 2
         class_label  = Species) %>%
  select(contains("feat"), class_num, class_label)
nn_dat %>% head(3)
```

    ## # A tibble: 3 x 6
    ##   sepal_l_feat[,1] sepal_w_feat[,1] petal_l_feat[,1] petal_w_feat[,1] class_num
    ##              <dbl>            <dbl>            <dbl>            <dbl>     <dbl>
    ## 1           -0.898            1.02             -1.34            -1.31         0
    ## 2           -1.14            -0.132            -1.34            -1.31         0
    ## 3           -1.38             0.327            -1.39            -1.31         0
    ## # … with 1 more variable: class_label <fct>

Then, we split the iris data into a training and a test data set,
setting aside 20% of the data for left out data partition, to be used
for final performance evaluation:

``` r
set.seed(2021)
test_f = 0.20
nn_dat = nn_dat %>%
  mutate(partition = sample(c('train','test'), nrow(.), replace = TRUE, 
                            prob = c(1 - test_f, test_f)))
```

Based on the partition, we can now create training and test data

``` r
x_train = nn_dat %>% filter(partition == 'train') %>% select(contains("feat")) %>% as.matrix
y_train = nn_dat %>% filter(partition == 'train') %>% pull(class_num) %>% to_categorical(3)
x_test  = nn_dat %>% filter(partition == 'test')  %>% select(contains("feat")) %>% as.matrix
y_test  = nn_dat %>% filter(partition == 'test')  %>% pull(class_num) %>% to_categorical(3)
```

## Set Architecture

With the data in place, we now set the architecture of our artificical
neural network:

``` r
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 4, activation = 'relu', input_shape = 4) %>% 
  layer_dense(units = 3, activation = 'softmax')
model %>% summary
```

    ## Model: "sequential"
    ## ________________________________________________________________________________
    ## Layer (type)                        Output Shape                    Param #     
    ## ================================================================================
    ## dense_1 (Dense)                     (None, 4)                       20          
    ## ________________________________________________________________________________
    ## dense (Dense)                       (None, 3)                       15          
    ## ================================================================================
    ## Total params: 35
    ## Trainable params: 35
    ## Non-trainable params: 0
    ## ________________________________________________________________________________

As expected we see 35 trainable parameters. Next, the architecture set
in the model needs to be compiled:

``` r
model %>% compile(
  loss      = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics   = c('accuracy')
)
```

## Train the Artificial Neural Network

Lastly we fit the model and save the training progress in the `history`
object:

``` r
history <- fit(model,
               x = x_train, y = y_train,
               epochs           = 200,
               batch_size       = 20,
               validation_split = 0,
               verbose = F)
plot(history)
```

![](tensorflow_tutorial_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

## Evaluate Network Performance

The final performance can be obtained like so:

``` r
perf = model %>% evaluate(x_test, y_test)
print(perf)
```

    ##      loss  accuracy 
    ## 0.1480854 1.0000000

Then we can augment the `nn_dat` for plotting:

``` r
plot_dat = nn_dat %>% filter(partition == 'test') %>%
  mutate(class_num = factor(class_num),
         y_pred    = factor(predict_classes(model, x_test)),
         Correct   = factor(ifelse(class_num == y_pred, "Yes", "No")))
plot_dat %>% select(-contains("feat")) %>% head(3)
```

    ## # A tibble: 3 x 5
    ##   class_num class_label partition y_pred Correct
    ##   <fct>     <fct>       <chr>     <fct>  <fct>  
    ## 1 0         setosa      test      0      Yes    
    ## 2 0         setosa      test      0      Yes    
    ## 3 0         setosa      test      0      Yes

and lastly, we can visualize the confusion matrix like so:

``` r
title     = "Classification Performance of Artificial Neural Network"
sub_title = str_c("Accuracy = ", round(perf["accuracy"], 3) * 100, "%")
x_lab     = "True iris class"
y_lab     = "Predicted iris class"
ggplot(plot_dat, aes(x = class_num, y = y_pred, colour = Correct)) +
  geom_jitter(height = 0.25, width = 0.25) +
  scale_x_discrete(labels = levels(nn_dat$class_label)) +
  scale_y_discrete(labels = levels(nn_dat$class_label)) +
  theme_bw() +
  labs(title = title, subtitle = sub_title, x = x_lab, y = y_lab)
```

![](tensorflow_tutorial_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

## Conclusion

Here, we created a 3-class predictor with an accuracy of 100% on a left
out data partition. I hope this little post illustrated how you can get
started building artificial neural network using [Keras and TensorFlow
in R](https://keras.rstudio.com/). This was a basic minimal example. It
should be noted that the network can be expanded to create full deep
Learning networks and furthermore, the entire TensorFlow API is
available. It also goes to show how important it is for a data
scientist, that the tools needed to go efficiently from idea to
implementation is available - Available and accessible technology is the
cornerstone of modern data science.

# Image Classifier - Hand-written numbers

This first example is from Rstudio tensorflow’s package presentation
[site](https://tensorflow.rstudio.com/tutorials/beginners/)

Let’s start by loading and preparing the MNIST dataset. The values of
thee pixels are integers between 0 and 255 and we will convert them to
floats between 0 and 1.

``` r
mnist <- dataset_mnist()
mnist$train$x <- mnist$train$x/255
mnist$test$x <- mnist$test$x/255
```

Now, let’s define the a Keras model using the sequential API.

``` r
model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(28, 28)) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(0.2) %>% 
  layer_dense(10, activation = "softmax")
```

**Note** that when using the Sequential API the first layer must specify
the input\_shape argument which represents the dimensions of the input.
In our case, images 28x28.

After defining the model, you can see information about layers, number
of parameters, etc with the summary function:

``` r
summary(model)
```

    ## Model: "sequential_1"
    ## ________________________________________________________________________________
    ## Layer (type)                        Output Shape                    Param #     
    ## ================================================================================
    ## flatten (Flatten)                   (None, 784)                     0           
    ## ________________________________________________________________________________
    ## dense_3 (Dense)                     (None, 128)                     100480      
    ## ________________________________________________________________________________
    ## dropout (Dropout)                   (None, 128)                     0           
    ## ________________________________________________________________________________
    ## dense_2 (Dense)                     (None, 10)                      1290        
    ## ================================================================================
    ## Total params: 101,770
    ## Trainable params: 101,770
    ## Non-trainable params: 0
    ## ________________________________________________________________________________

The next step after building the model is to compile it. It’s at compile
time that we define what loss will be optimized and what optimizer will
be used. You can also specify metrics, callbacks and etc that are meant
to be run during the model fitting.

Compiling is done with the compile function:

``` r
model %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )
```

Note that compile and fit (which we are going to see next) modify the
model object in place, unlike most R functions.

Now let’s fit our model:

``` r
model %>% 
  fit(
    x = mnist$train$x, y = mnist$train$y,
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )
```

We can now make predictions with our model using the predict function:

``` r
predictions <- predict(model, mnist$test$x)
head(predictions, 2)
```

    ##              [,1]         [,2]         [,3]         [,4]         [,5]
    ## [1,] 1.801528e-07 3.383665e-09 6.293833e-06 1.461414e-04 1.546005e-10
    ## [2,] 8.560576e-08 1.359207e-04 9.997715e-01 9.090851e-05 9.306498e-12
    ##              [,6]         [,7]         [,8]         [,9]        [,10]
    ## [1,] 1.921820e-07 2.801314e-12 9.998441e-01 2.052107e-07 2.877850e-06
    ## [2,] 5.400724e-08 2.413190e-07 1.503159e-10 1.185691e-06 1.127817e-10

By default predict will return the output of the last Keras layer. In
our case this is the probability for each class. You can also use
predict\_classes and predict\_proba to generate class and probability -
these functions are slightly different then predict since they will be
run in batches.

You can access the model performance on a different dataset using the
evaluate function, for example:

``` r
model %>% 
  evaluate(mnist$test$x, mnist$test$y, verbose = 0)
```

    ##       loss   accuracy 
    ## 0.08321846 0.97430003

Our model achieved \~90% accuracy on the test set.

Unlike models built with the lm function, to save Keras models for later
prediction, you need to use specialized functions, like save\_model\_tf:

``` r
save_model_tf(object = model, filepath = "model")
```

You can then reload the model and make predictions with:

``` r
reloaded_model <- load_model_tf("model")
all.equal(predict(model, mnist$test$x), predict(reloaded_model, mnist$test$x))
```

    ## [1] TRUE

# Notebook info

## Warnings

``` r
warnings()
```

## Session Info

``` r
sessionInfo()
```

    ## R version 4.0.0 (2020-04-24)
    ## Platform: x86_64-pc-linux-gnu (64-bit)
    ## Running under: CentOS Linux 7 (Core)
    ## 
    ## Matrix products: default
    ## BLAS/LAPACK: /home/labs/schwartzlab/miguelg/.local/share/r-miniconda/envs/r-reticulate/lib/libopenblasp-r0.3.10.so
    ## 
    ## locale:
    ##  [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C              
    ##  [3] LC_TIME=en_US.UTF-8        LC_COLLATE=en_US.UTF-8    
    ##  [5] LC_MONETARY=en_US.UTF-8    LC_MESSAGES=en_US.UTF-8   
    ##  [7] LC_PAPER=en_US.UTF-8       LC_NAME=C                 
    ##  [9] LC_ADDRESS=C               LC_TELEPHONE=C            
    ## [11] LC_MEASUREMENT=en_US.UTF-8 LC_IDENTIFICATION=C       
    ## 
    ## attached base packages:
    ##  [1] stats4    grid      parallel  stats     graphics  grDevices utils    
    ##  [8] datasets  methods   base     
    ## 
    ## other attached packages:
    ##  [1] forcats_0.5.0               stringr_1.4.0              
    ##  [3] dplyr_1.0.2                 purrr_0.3.4                
    ##  [5] readr_1.4.0                 tidyr_1.1.2                
    ##  [7] tibble_3.0.4                tidyverse_1.3.0            
    ##  [9] keras_2.3.0.0               rtracklayer_1.48.0         
    ## [11] GenomicAlignments_1.24.0    Rsamtools_2.4.0            
    ## [13] Biostrings_2.56.0           XVector_0.28.0             
    ## [15] SummarizedExperiment_1.18.2 DelayedArray_0.14.1        
    ## [17] matrixStats_0.57.0          Biobase_2.48.0             
    ## [19] GenomicRanges_1.40.0        GenomeInfoDb_1.24.2        
    ## [21] IRanges_2.22.2              S4Vectors_0.26.1           
    ## [23] BiocGenerics_0.34.0         RColorBrewer_1.1-2         
    ## [25] EnvStats_2.3.1              reshape2_1.4.4             
    ## [27] gtools_3.8.2                ggplot2_3.3.2              
    ## [29] plyr_1.8.6                  optparse_1.6.6             
    ## [31] magrittr_2.0.1             
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] nlme_3.1-150           bitops_1.0-6           fs_1.5.0              
    ##  [4] lubridate_1.7.9.2      httr_1.4.2             tools_4.0.0           
    ##  [7] backports_1.1.10       utf8_1.1.4             R6_2.5.0              
    ## [10] mgcv_1.8-33            DBI_1.1.0              colorspace_2.0-0      
    ## [13] withr_2.3.0            tidyselect_1.1.0       compiler_4.0.0        
    ## [16] cli_2.1.0              rvest_0.3.6            xml2_1.3.2            
    ## [19] labeling_0.4.2         scales_1.1.1           rappdirs_0.3.1        
    ## [22] tfruns_1.4             digest_0.6.27          rmarkdown_2.5         
    ## [25] base64enc_0.1-3        pkgconfig_2.0.3        htmltools_0.5.0       
    ## [28] dbplyr_2.0.0           rlang_0.4.8            readxl_1.3.1          
    ## [31] rstudioapi_0.13        farver_2.0.3           generics_0.1.0        
    ## [34] jsonlite_1.7.1         tensorflow_2.2.0       BiocParallel_1.22.0   
    ## [37] RCurl_1.98-1.2         GenomeInfoDbData_1.2.3 Matrix_1.2-18         
    ## [40] fansi_0.4.1            Rcpp_1.0.5             munsell_0.5.0         
    ## [43] reticulate_1.18        lifecycle_0.2.0        stringi_1.5.3         
    ## [46] whisker_0.4            yaml_2.2.1             zlibbioc_1.34.0       
    ## [49] crayon_1.3.4           lattice_0.20-41        splines_4.0.0         
    ## [52] haven_2.3.1            hms_0.5.3              zeallot_0.1.0         
    ## [55] knitr_1.30             pillar_1.4.6           reprex_0.3.0          
    ## [58] XML_3.99-0.5           glue_1.4.2             evaluate_0.14         
    ## [61] modelr_0.1.8           vctrs_0.3.4            cellranger_1.1.0      
    ## [64] gtable_0.3.0           getopt_1.20.3          assertthat_0.2.1      
    ## [67] xfun_0.19              broom_0.7.2            ellipsis_0.3.1

## Time to knit

``` r
tEnd <- Sys.time()
tDif <- tEnd - tStart
cat("Time to knit notebook:", round(tDif, 2), units(tDif))
```

    ## Time to knit notebook: 35.09 secs
