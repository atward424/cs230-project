library(data.table)
library(dtplyr)

smeta = metadata[ order(metadata$non_na, decreasing=T), ]

namess = rownames(smeta)[1:15]

names2 = gsub(" ", "\\.", namess )

cnames = colnames(out.patients.df)

colstouse = rep(0, length(cnames))
for (ind in 1:length(colstouse)) {
  for (jind in 1:length(names2)) {
    if (grepl(names2[jind], cnames[ind])) {
      colstouse[ind] = 1
      print(c(ind, names2[jind], cnames[ind]))
      if (colstouse[3]==1) {
        print(c("here", ind, jind, names2[jind], cnames[ind]))
      }
      break
    }
  }
}
sum(colstouse)


feats = out.patients.df[!is.na(patoutcomes), which(colstouse==1)]

patcsns = out.patients.df[!is.na(patoutcomes), 1]
times2 = out.patients.df[!is.na(patoutcomes), 2]
p = dim(feats)[2]
mp = ceiling(p/3)


disctimes = patoutcomes[!is.na(patoutcomes)]

diffdiscs = as.numeric(difftime(disctimes, times2, units = "secs"))

disc.before.noon = as.numeric(diffdiscs < 4*60*60)
disc.24.hr = as.numeric(diffdiscs < 24*60*60)

feats2 = data.table(feats)
feats2 = feats2[, ADT_DEPARTMENT_NAME := NULL]
feats2 = feats2[, R.CARDIAC.RHYTHM := NULL]
feats2 = feats2[, R.PAIN.ASSESSMENT := NULL]
feats2 = feats2[, R.RT.ICU.O2.THERAPY := NULL]



datt = rfImpute(feats2, disc.24.hr, iter = 10, ntree = 501)
rf.fitsty = randomForest(disc.24.hr ~ ., data = datt, #x = feats, y = disc.24.hr, 
                         ntree = 501, importance = TRUE, mtry = mp, na.action = na.roughfix)



preds = rf.fitsty$predicted

sscurve = data.frame(thres = numeric(),
                     ac = numeric(), 
                     sens = numeric(),
                     spec = numeric())
for (thres in (0:50)*0.002) {
  # thres = 0.04
  acc = sum((preds > thres) == disc.24.hr)/length(preds)
  sens = sum(preds[disc.24.hr==1] > thres)/length(preds[disc.24.hr==1])
  spec=sum(preds[disc.24.hr==0] <= thres)/length(preds[disc.24.hr==0])
  
  temp=  data.frame(thres, acc, sens, spec)
  sscurve = rbind(sscurve, temp)
}

View(sscurve)
