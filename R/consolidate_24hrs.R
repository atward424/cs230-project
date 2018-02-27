setwd("C:\\Users\\Andrew\\Documents\\Stanford\\mse463\\code")

#### get one data point for every 24 hours, 8am to 8pm ------
# load("patient_relevant_data.Rda")
# readRDS("patient_relevant_data2.RDS")

# split up numbers and strings
pat.fs.numbers = pat.fs.df[, as.logical(is.number)]
pat.fs.numbers = pat.fs.numbers [, -1]
pat.fs.strings = pat.fs.df[, !as.logical(is.number)]
pat.fs.strings = pat.fs.strings[, -1]










rm(out.patients.df)









count = 0
last_csn_ind = 1
last_csn_ind = which(last_csn == unique(pat.fs.df$PAT_ENC_CSN_ID))
for (csn in unique(pat.fs.df$PAT_ENC_CSN_ID)[(last_csn_ind+1):length(unique(pat.fs.df$PAT_ENC_CSN_ID))]) { 
  count = count+1
  if (count %% 50 == 0) {
    print(paste("iter", as.character(count), "out of", as.character(length(unique(pat.fs.df)))))
  }
  rm(out.patient.df)
  patient.df = pat.fs.df[pat.fs.df$PAT_ENC_CSN_ID == csn, ]
  patient.strs = pat.fs.strings[pat.fs.df$PAT_ENC_CSN_ID == csn, ]
  patient.strs.filled = na.locf(patient.strs)
  patient.nums = pat.fs.numbers[pat.fs.df$PAT_ENC_CSN_ID == csn, ]
  
  mintime = min(patient.df$RECORDED_TIME)
  maxtime = max(patient.df$RECORDED_TIME)
  
  # get the first day to use
  firstday = mintime
  firstday$mday = firstday$mday + 1
  firstday$hour = 8
  firstday$min = 0
  firstday$sec = 0
  
  lastday = maxtime
  lastday$hour = 8
  lastday$min = 0
  lastday$sec = 0
  
  diffdays = lastday - firstday
  numdays = as.numeric(diffdays)
  
  if (is.na(numdays)) {
    next
  }
  if (numdays < 1) {
    next
  }
  
  # out.patient.strs = pat.fs.strings[1,]
  
  # day = 1
  for (day in 1:numdays) {
    
    thisday = firstday
    thisday$mday  = thisday$mday + (day - 1)
    
    prevday = thisday
    prevday$mday = thisday$mday - 1
    
    out.patient.row = data.frame(patient_csn = csn, 
                                 time=thisday)
    
    
    
    inds_to_consider = which((patient.df$RECORDED_TIME < thisday) & (patient.df$RECORDED_TIME >= prevday))
    if (length(inds_to_consider) == 0) {
      print(paste("no data for", as.character(csn), "on day", as.character(thisday)))
      next
    }
    out.patient.strs = patient.strs[max(inds_to_consider), ]
    
    # col = 1
    out.patient.nums = data.frame()
    for (col in 1:length(patient.nums)) {
      to_consider = patient.nums[inds_to_consider, col]
      to_consider = as.numeric(to_consider)
      not_nas = which(!is.na(to_consider))
      name = colnames(patient.nums)[col]
      
      out.names = c(paste0('avg.', name), 
                    paste0('min.', name), 
                    paste0('max.', name), 
                    paste0('sd.', name), 
                    # paste('aml.', name), 
                    paste0('fml.', name)
      )
      
      if (length(not_nas) == 0) {
        temp.nums = as.numeric(rep(NA, length(out.names)))
      } else {
        temp.nums = as.numeric(c(mean(to_consider, na.rm = T),
                                 min(to_consider, na.rm = T),
                                 max(to_consider, na.rm = T),
                                 sd(to_consider, na.rm = T),
                                 # admission value - to_consider[max(not_nas)]
                                 to_consider[min(not_nas)] - to_consider[max(not_nas)]
        ))
        print(temp.nums)
      }
      num.df = data.frame(t(temp.nums))
      colnames(num.df) = out.names
      out.patient.row = cbind(out.patient.row, num.df)
    }
    
    out.patient.row = cbind(out.patient.row, out.patient.strs)
    
    if (!exists('out.patient.df')) {
      out.patient.df = out.patient.row
    } else {
      out.patient.df = rbind(out.patient.df, out.patient.row)
    }
    
  }
  
  if (!exists('out.patients.df')) {
    out.patients.df = out.patient.df
  } else {
    if (exists('out.patient.df')) {
      out.patients.df = rbind(out.patients.df, out.patient.df)
    }
  }
  
}

out.pat.df.test2 = out.patients.df
last_csn = csn

length(unique(out.patients.df$patient_csn))

day.num.patients = length(unique(out.pat.df.test2$patient_csn))

print(paste("number of patients in patient/day:", day.num.patients))
print(paste("total patient days:", length(out.pat.df.test2$patient_csn)))
day.percents = colSums(!is.na(out.pat.df.test2))/dim(out.pat.df.test2)[1]
day.metadata = data.frame(variable = colnames(out.pat.df.test2),  non_na = round(day.percents, 3))
# day.metadata = day.metadata[, -1]
View(day.metadata)

nvars = length(out.patients.df)
sum(!is.na(out.patients.df[,3:nvars]))/prod(dim(out.patients.df))
View(out.patients.df)


write.csv(metadata, "percentage_of_non_nas.csv")

write.table(out.patients.df, file = "outPatientsDf.csv")

saveRDS(out.patients.df, "patients_per_24hrs.RDS")



# names(unclass(mintime))

# split_into_rows = function(csn, pat.fs.df) {
#   patient.df = pat.fs.df[pat.fs.df$PAT_ENC_CSN_ID == csn, ]
#   # sum(!is.na(patient.df))/prod(dim(patient.df))
#   out.patient.df = 
# }

