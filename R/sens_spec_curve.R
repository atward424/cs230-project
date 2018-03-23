library(ggplot2)

setwd("C:\\Users\\Andrew\\Documents\\Stanford\\cs230\\cs230-code-examples\\pytorch\\vision")

sweep = function(preds, labs) {
  
  ss_data = data.frame(thres = numeric(),
                            ac = numeric(), 
                            sens = numeric(),
                            spec = numeric())
  for (thres in (0:1000)*0.001) {
    # thres = 0.04
    acc = sum((preds > thres) == labs)/length(preds)
    sens = sum(preds[labs==1] > thres)/length(preds[labs==1])
    spec=sum(preds[labs==0] <= thres)/length(preds[labs==0])
    
    temp=  data.frame(thres, acc, sens, spec)
    ss_data = rbind(ss_data, temp)
  }

  return(ss_data)
}


# tr_preds = read.csv('outputs_lstm\\train_predictions.csv')
# tr_labs = read.csv('outputs_lstm\\train_labels.csv')
# val_preds = read.csv('outputs_lstm\\val_predictions.csv')
# val_labs = read.csv('outputs_lstm\\val_labels.csv')
# ts_preds = read.csv('outputs_lstm\\test_predictions.csv')
# ts_labs = read.csv('outputs_lstm\\test_labels.csv')
# factor = 1e-1

tr_preds = read.csv('outputs\\trainf_predictions.csv')
tr_labs = read.csv('outputs\\trainf_labels.csv')
val_preds = read.csv('outputs\\valf_predictions.csv')
val_labs = read.csv('outputs\\valf_labels.csv')
ts_preds = read.csv('outputs\\testf_predictions.csv')
ts_labs = read.csv('outputs\\testf_labels.csv')
factor = 2e1

train_ss = sweep(tr_preds$X0*factor, tr_labs$X0)
val_ss = sweep(val_preds$X0*factor, val_labs$X0)
test_ss = sweep(ts_preds$X0*factor, ts_labs$X0)


# tr_labs = read.csv('data\\train_outcomes.csv')
# val_labs = read.csv('data\\val_outcomes.csv')
# ts_labs = read.csv('data\\test_outcomes.csv')
# tr_preds = rf.disc24h$predicted
# val_preds = rf.disc24h$test$predicted
# ts_preds = rf.disc24htest$test$predicted

# train_ss = sweep(tr_preds, tr_labs$disc.24.hr)
# val_ss = sweep(val_preds, val_labs$disc.24.hr)
# test_ss = sweep(ts_preds, ts_labs$disc.24.hr)

# write.csv(tr_preds, 'rf_train_preds.csv', row.names = F)
# write.csv(val_preds, 'rf_val_preds.csv', row.names = F)
# write.csv(ts_preds, 'rf_test_preds.csv', row.names = F)



comb = rbind(cbind(train_ss, type=rep('train', dim(train_ss)[1])), 
             cbind(val_ss, type=rep('val', dim(val_ss)[1])),
             cbind(test_ss, type=rep('test', dim(test_ss)[1])))

comb$spec = 1-comb$spec

gg = ggplot(data = comb, aes(spec, sens)) + geom_point(aes(color = type)) + theme_bw()+ xlim(0,1) + ylim(0,1) + 
  scale_color_manual(values=c("red", "blue", "green"))+
  labs(title = paste("LSTM Model"), x=paste("False Positive Rate", ""), y = "True Positive Rate") +
  theme(legend.title = element_blank()) + theme(legend.position = c(0.8, 0.3))
print(gg)
# gg = ggplot(data = comb, aes(spec, sens)) + geom_point(aes(color = type)) + theme_bw()+ xlim(0,1) + ylim(0,1) + 
#   scale_color_manual(values=c("red", "blue", "green"))+
#   labs(title = paste("Baseline Model"), x=paste("Sensitivity", ""), y = "Specificity") +
#   theme(legend.title = element_blank()) + theme(legend.position = c(0.1, 0.3))
# print(gg)