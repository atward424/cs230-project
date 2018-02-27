library(vtreat)
setwd("C:\\Users\\Andrew\\Documents\\Stanford\\mse463\\code")

# read in patient discharge data
tdat = read.csv("Patient_transfer_out_data.csv")
hfd = tdat[tdat$Dept.Abbrev %in% c("2NCVICU", "3W", "PCU374"), ] # for some reason, this gets rid of 115 patients
hfd = hfd[hfd$Hospital.Discharge.Dt.Tm != "", ]
hfd$Hospital.Discharge.Dt.Tm = as.character(hfd$Hospital.Discharge.Dt.Tm)

# read in all features
out.patients.df = read.csv("outPatientsDf2.csv", header = T,sep = ",")

# check for patients without discharge times
patoutcomes = sapply(out.patients.df$patient_csn, function(x) {
  bb = unique(hfd$Hospital.Discharge.Dt.Tm[hfd$Primary.CSN == x])
  if (length(bb) == 1)
    return(bb)
  return(NA)
})
patoutcomes = strptime(patoutcomes, format = "%Y/%m/%d %H:%M:%S")
sum(is.na(patoutcomes))/length(patoutcomes)



# load in numeric features
feats_numeric = out.patients.df[!is.na(patoutcomes), 4:243]

# load in categorical features
feats_cat = out.patients.df[!is.na(patoutcomes), 244:dim(out.patients.df)[2]]

# replace categorical features with more than 53 levels to binary
uniks = apply(feats_cat, 2, unique)
luniks = lengths(uniks)
feats_cat = droplevels(feats_cat)
feats_cat[feats_cat ==""] <- NA
for (col in which(luniks >= 53)) {
  feats_cat[, col] = as.numeric(!is.na(feats_cat[, col]))
}
feats_cat = droplevels(feats_cat)

# change the blood pressure fields into numeric fields
format_bp = function(x, title) {
  x2 = sapply(x, function(y) {
    if (is.na(y)) return(c(NA, NA, NA))
    bp = as.numeric(strsplit(as.character(y), '/')[[1]])
    return(c(bp[1], bp[2], bp[1]/(bp[2]+0.00001)))
  })
  x3 = data.frame(t(x2))
  colnames(x3) = c(paste0(title, '.SYSTOLIC'), 
                   paste0(title, '.DIASTOLIC'), 
                   paste0(title, '.RATIO'))
  return(x3)
}

bp = format_bp(feats_cat$BLOOD.PRESSURE, 'BP')
abp = format_bp(feats_cat$R.ARTERIAL.LINE.BLOOD.PRESSURE, 'ABP')

feats_numeric = cbind(feats_numeric, bp, abp)

# load discharge times and create outcome variable
disctimes = patoutcomes[!is.na(patoutcomes)]
times3 = out.patients.df[!is.na(patoutcomes), 3]
diffdiscs = as.numeric(difftime(disctimes, times3, units = "secs"))
disc.before.noon = as.numeric(diffdiscs < 4*60*60)
disc.24.hr = as.numeric(diffdiscs < 24*60*60)
disc.7.weeks = as.numeric(diffdiscs < 7*7*24*60*60)




# create the full feature matrix
combined_features = data.frame(feats_numeric, feats_cat, check.names=T)

# remove "expected discharge date" from features
combined_features = combined_features[, c(-339, -340)]

# create an outcome matrix
cleaned_outcomes = as.matrix(data.frame(cbind(csn=out.patients.df$patient_csn[!is.na(patoutcomes)],
                                              disc.before.noon=disc.before.noon,
                                              disc.24.hr=disc.24.hr,
                                              disc.7.weeks = disc.7.weeks,
                                              diffdiscs = diffdiscs)))

# generate summary histogram for patient length of stays
length_of_stays = as.numeric(table(cleaned_outcomes[,1]))
length_of_stay_csns = unique(cleaned_outcomes[,1])
qplot(as.numeric(length_of_stays), main = paste('Length of Stay Distribution,', "N = ", as.character(length(length_of_stays))),
      geom = "histogram", , fill = "blue", bins=100) + theme_bw()  + theme(legend.position = "none") + 
  labs(x='LOS (days)', y = "number of patients") + xlim(0, 200)
largest_los = max(as.numeric(length_of_stays))

# split into train-val-test, while keeping distribution relatively consistent
long_los = which(length_of_stays > 60)
med_los = which(length_of_stays > 14 & length_of_stays <= 60)
short_los = which(length_of_stays <= 14)

# Compute sample sizes.
trsplit   = 0.6 
vsplit    = 0.2 
tssplit   = 0.2 

# Create the randomly-sampled indices for the long_los, med_los, and short_los
# patients. Use setdiff() to avoid overlapping subsets of indices.
trl    = sort(sample(long_los, size=floor(trsplit*length(long_los))))
not_trl = setdiff(long_los, trl)
vall = sort(sample(not_trl, size=floor(vsplit*length(long_los))))
testl = setdiff(not_trl, vall)

trm    = sort(sample(med_los, size=floor(trsplit*length(med_los))))
not_trm = setdiff( med_los, trm)
valm = sort(sample(not_trm, size=floor(vsplit*length(med_los))))
testm = setdiff(not_trm, valm)

trs    = sort(sample(short_los, size=floor(trsplit*length(short_los))))
not_trs = setdiff( short_los, trs)
vals = sort(sample(not_trs, size=floor(vsplit*length(short_los))))
tests = setdiff(not_trs, vals)

# subset out the training, val, and test dataframes
training_csns = length_of_stay_csns[c(trl, trm, trs)]
training_features = combined_features[cleaned_outcomes[,1] %in% training_csns, ]
training_outcomes = cleaned_outcomes[cleaned_outcomes[,1] %in% training_csns, ]

val_csns = length_of_stay_csns[c(vall, valm, vals)]
val_features = combined_features[cleaned_outcomes[,1] %in% val_csns, ]
val_outcomes = cleaned_outcomes[cleaned_outcomes[,1] %in% val_csns, ]

test_csns = length_of_stay_csns[c(testl, testm, tests)]
test_features = combined_features[cleaned_outcomes[,1] %in% test_csns, ]
test_outcomes = cleaned_outcomes[cleaned_outcomes[,1] %in% test_csns, ]


# create cleaned feature matrices with one-hotted categorical variables
vars_to_one_hot = colnames(combined_features)
tplan <- vtreat::designTreatmentsZ(training_features, vars_to_one_hot, 
                                   minFraction= 0,
                                   verbose=TRUE)
sf <- tplan$scoreFrame
newvars <- sf$varName[sf$code %in% c("lev", "clean", "isBAD")] 
train_features_cleaned <- as.matrix(vtreat::prepare(tplan, training_features, 
                                                    varRestriction = newvars))
print(dim(train_features_cleaned))
val_features_cleaned <- as.matrix(vtreat::prepare(tplan, val_features, 
                                                    varRestriction = newvars))
print(dim(val_features_cleaned))
test_features_cleaned <- as.matrix(vtreat::prepare(tplan, test_features, 
                                                    varRestriction = newvars))
print(dim(test_features_cleaned))




# save all features and outcomes to csvs
write.csv(train_features_cleaned, file='train_features_onehot.csv', row.names = F)
write.csv(training_outcomes, file='train_outcomes.csv', row.names = F)

write.csv(val_features_cleaned, file='val_features_onehot.csv', row.names = F)
write.csv(val_outcomes, file='val_outcomes.csv', row.names = F)

write.csv(test_features_cleaned, file='test_features_onehot.csv', row.names = F)
write.csv(test_outcomes, file='test_outcomes.csv', row.names = F)
