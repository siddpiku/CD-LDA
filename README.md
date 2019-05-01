# CD-LDA
This is an implementation of the paper `Learning Latent Events from Network Message Logs' for a synthetic dataset. It deals with finding patterns in mixture of time series

datainitCDLDA.py - generates a time series data
 
CDLDA.py - runs the CDLDA algorithm. Note that it has options to use different metrics for change detection and different algorithms for LDA. For using the spectral_lda algorithm you need to download this code (https://github.com/Mega-DatA-Lab/SpectralLDA) . There are some hyper parameters that one needs to set for the LDA algorithm which varies according to the data. If you are using the Gibbs sampling version of LDA you need to set the n_topics variable to the number of topics.

For any questions regarding the implementation please contact the author at sidd.piku@gmail.com
