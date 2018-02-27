setwd("C:\\Users\\Andrew\\Documents\\Stanford\\mse463\\code")
library(dplyr)

library(tidyr)

library(zoo)

# read in data
load("full_data.Rdata")
flo.rows = read.csv("flo_rows_charles_05242017.csv", header = TRUE)

# read in census data
adt = readRDS("ADT_Census_Data_deidentified.RDS")
bpdat = readRDS("Blood_Pressure_CVICU_PCU374_20150101_20160901.RDS")


# how many patients do we have?
print(paste("number of patients:", length(unique(fs.data$PAT_ENC_CSN_ID))))

# get all the variables (in/out) that we care about
names_to_use_inds = which(flo.rows$is_input == 1 | flo.rows$is_output == 1)
names_to_use = as.character(flo.rows$name[names_to_use_inds])
input_inds = which(flo.rows$is_input == 1)
input_fields = as.character(flo.rows$name[input_inds])
output_inds = which(flo.rows$is_output == 1)
output_fields = as.character(flo.rows$name[output_inds])

# # alternate data read-in (data Isabel found talking with nurses)
info = read.csv("Request_DataDump.csv", header=TRUE)
names.lots.list = info$Possible.Flowsheet.Measure.Name
names.lots = unlist(lapply(as.character(names.lots.list), strsplit, ", +"))
names.lots = gsub("\n", "", names.lots)

# get alternate variables we care about
names.in = names.lots[names.lots %in% unique(fs.data$FLO_ROW_NAME)]
names.in[length(names.in)+1] = "R IP DATE OF DEATH"

# Which variables to get? (uncomment one)
# fs.data.relevant = fs.data[fs.data$FLO_ROW_NAME %in% names_to_use, ] # Charles marked
# fs.data.relevant = fs.data[fs.data$FLO_ROW_NAME %in% names.in, ] # isabel found with nurses
fs.data.relevant = fs.data[fs.data$FLO_ROW_NAME %in% c(names_to_use, names.in), ] # both

# remove unnecessary columns and format data correctly
fs.data.relevant.d = fs.data.relevant[, c(-3, -4, -5, -6, -8, -11:-16)]

#View duplicated rows
fs.data.relevant.d = fs.data.relevant.d[order(fs.data.relevant.d[,1], fs.data.relevant.d[,3]), ]
alldups = duplicated(fs.data.relevant.d[, 1:4]) | duplicated(fs.data.relevant.d[,1:4], fromLast = T)
# View(fs.data.relevant.d[alldups,])

# get rid of all duplicated rows
dups = duplicated(fs.data.relevant.d[, 1:4])
fs.data.relevant.d.nodups = fs.data.relevant.d[!dups, ]
fs.data.wide.d <- spread(fs.data.relevant.d.nodups, FLO_ROW_NAME, FLO_VALUE)
fs.data.wide.dd = fs.data.wide.d[order(fs.data.wide.d[,1], fs.data.wide.d[,3]), ]

# get rid of all duplicated rows in ALL OF DATA
# afs.data.relevant.d = fs.data[, c(-3, -4, -5, -6, -8, -11:-16)]
# adups = duplicated(afs.data.relevant.d[, 1:4])
# afs.data.relevant.d.nodups = afs.data.relevant.d[!adups, ]
# afs.data.wide.d <- spread(afs.data.relevant.d.nodups, FLO_ROW_NAME, FLO_VALUE)
# afs.data.wide.dd = afs.data.wide.d[order(afs.data.wide.d[,1], afs.data.wide.d[,3]), ]

# how many patients do we have? View the data frame
print(paste("patients to analyze:", length(unique(fs.data.wide.dd$PAT_ENC_CSN_ID))))
# What percentage of fields are NA?
# print(paste("percentage of NAs in data:", sum(is.na(fs.data.wide.dd))/prod(dim(fs.data.wide.dd))))
# View(fs.data.wide.dd[1:100,])

# turn numeric character fields into numbers
pat.fs.data = fs.data.wide.dd
st = proc.time()
a=lapply(fs.data.wide.dd, function(col, pat.fs.data) {
  col.index  = which(colnames(pat.fs.data) == colnames(col))
  col[col == ""] = NA
  if (suppressWarnings(all(!is.na(as.numeric(as.character(col[!is.na(col)])))))) {
    pat.fs.data[!is.na(col), col.index] = as.numeric(as.character(col[!is.na(col)]))
  } else {
    pat.fs.data[, col.index] = col
  }
}, pat.fs.data)
rm(a)
# print(proc.time()-st)

# see which fields are numeric
is.number <- lapply(fs.data.wide.dd, function(col) {
  col[col == ""] = NA
  if (suppressWarnings(all(!is.na(as.numeric(as.character(col[!is.na(col)])))))) {
    return(1)
  } else {
    return(0)
  }
})
is.number = unlist(is.number)

# how many numeric columns do we have?
print(paste("numeric fields:", sum(is.number)))
print(paste("non-numeric fields:", sum(!is.number)))

# clean up and reorder data 
pat.fs.df = as.data.frame(pat.fs.data)
colnames(pat.fs.df) = make.names(colnames(pat.fs.df))
pat.fs.df$RECORDED_TIME = strptime(pat.fs.df$RECORDED_TIME, format = "%Y-%m-%d %H:%M:%S")
pat.fs.df = pat.fs.df[, c(1,3,2,4:dim(pat.fs.df)[2])]
pat.fs.df = pat.fs.df[order(pat.fs.df[,1], pat.fs.df[,2]), ]
pat.fs.df = pat.fs.df[pat.fs.df$ADT_DEPARTMENT_NAME != 'PRE-ADMISSION', ]

nfields = length(pat.fs.df)
print(paste("% of non-NA values:", sum(!is.na(pat.fs.df[,4:nfields]))/prod(dim(pat.fs.df))))

# see how much of each variable is NA
percents = colSums(!is.na(pat.fs.df))/dim(pat.fs.df)[1]

metadata = data.frame(variable = colnames(pat.fs.df), is_numeric = is.number, non_na = round(percents, 3))
                      # is_input = (colnames(pat.fs.df) %in% input_fields), is_output = (colnames(pat.fs.df) %in% output_fields))
metadata = metadata[, -1]
View(metadata)

# save(pat.fs.df, file="patient_relevant_data.Rda")
# saveRDS(pat.fs.df, "patient_relevant_data.RDS")
# rm(list=setdiff(ls(), "pat.fs.df"))


