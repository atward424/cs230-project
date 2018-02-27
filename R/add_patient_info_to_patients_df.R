setwd("C:\\Users\\Andrew\\Documents\\Stanford\\mse463\\code")

full_census = read.csv("Census and Surgical Admits and Scheduled Date October 2017.csv")

tdat = read.csv("Patient_transfer_out_data.csv")

out.patients.df = read.csv("outPatientsDf.csv", header = T,sep = " ")
hfd = tdat[tdat$Dept.Abbrev %in% c("2NCVICU", "3W", "PCU374"), ] # for some reason, this gets rid of 115 patients
# hfd = tdat
hfd = hfd[hfd$Hospital.Discharge.Dt.Tm != "", ]
# View(hfd)


hfd$Hospital.Discharge.Dt.Tm = as.character(hfd$Hospital.Discharge.Dt.Tm)
# 
# hfd$Hospital.Discharge.Dt.Tm = strptime(hfd$Hospital.Discharge.Dt.Tm, format = "%Y/%m/%d %H:%M:%S")

patoutcomes = sapply(out.patients.df$patient_csn, function(x) {
  bb = unique(hfd$Hospital.Discharge.Dt.Tm[hfd$Primary.CSN == x])
  if (length(bb) == 1)
    return(bb)
  return(NA)
})

which(lengths(patoutcomes) > 1)
patoutcomes = strptime(patoutcomes, format = "%Y/%m/%d %H:%M:%S")
sum(is.na(patoutcomes))/length(patoutcomes)

inds_to_add = c(4, 6, 7, 9:19)

df.times = strptime(out.patients.df$time, format = "%Y-%m-%d_%H:%M:%S")
census.times = strptime(full_census$Effective.Date.Time, format = "%Y/%m/%d %H:%M:%S")


colnames(new_feats) = colnames(full_census)[inds_to_add]

relevant_census_inds = (census.times > strptime("2015-01-01", "%Y-%m-%d")) &
                        (census.times < strptime("2017-01-01", "%Y-%m-%d"))

rel_census = full_census[relevant_census_inds, ]
census.timesr = strptime(rel_census$Effective.Date.Time, format = "%Y/%m/%d %H:%M:%S")

new_feats = data.frame(rel_census[1:dim(out.patients.df)[1], inds_to_add])

i = 1
for (i in 10770:dim(out.patients.df)[1]) {
  ind = which((df.times[i] > census.timesr) & 
                (df.times[i] - 60*60*24 < census.timesr) & 
                (out.patients.df$patient_csn[i] == rel_census$Primary.CSN))
  if (length(ind) > 0) {
    new_feats[i, ] = rel_census[ind, inds_to_add]
  } else {
    new_feats[i, ] = NA
  }
  
  if ((i %% 10) == 0) {
    print(i)
  }
}

out.patients.df2 = cbind(out.patients.df, new_feats)

times = strptime(out.patients.df2$time, format='%Y-%m-%d_%H:%M:%S')
admit_times = strptime(out.patients.df2$Hospital.Admission.Dt.Tm, format='%Y/%m/%d %H:%M:%S')
time_in_hospital = as.numeric(times - admit_times)
day.of.year = as.numeric(format(times, "%j"))
month.of.year = format(times, "%b")

out.patients.df2$time = times
out.patients.df2$Hospital.Admission.Dt.Tm = admit_times
out.patients.df2['length.of.stay.hours'] = time_in_hospital
out.patients.df2['day.of.year'] = day.of.year
out.patients.df2['month.of.year'] = month.of.year


write.csv(out.patients.df2, file='outPatientsDf2.csv', sep = ',')
saveRDS(out.patients.df2, 'out.patients.df2.RDS')
