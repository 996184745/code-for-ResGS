rm(list = ls())
library(BGLR)

data_file = read.csv(file = "rice413/PhenoGenotype_Alkali spreading valueall.csv",
                     )


X=scale(wheat.X)
y=wheat.Y[,2]
n=length(y)

mae <- function(x,y) mean(abs(y - x))

# (4) Cross-validation using a loop
FoldNumber = 2

set.seed(666)
folds <- createFolds(y= y, k= FoldNumber)
yHatCV=rep(NA,n)

timeIn=proc.time()[3]
for(i in 1:FoldNumber){
  tst= folds[i]
  tst = unlist(tst)
  yNA=y
  yNA[tst]=NA
  fm=BGLR(y=yNA,ETA=list(list(X=X,model='BRR')),nIter=6000,burnIn=1000)
  yHatCV[tst]=fm$yHat[tst]
  mae(y[tst], fm$yHat[tst])
}

proc.time()[3]-timeIn


# for (i in 1:3)
# {
#   print(i)
# }


# folds=sample(1:5,size=n,replace=T)
# yHatCV=rep(NA,n)
# 
# timeIn=proc.time()[3]
# for(i in 1:max(folds)){
#   tst=which(folds==i)
#   yNA=y
#   yNA[tst]=NA
#   fm=BGLR(y=yNA,ETA=list(list(X=X,model='BRR')),nIter=6000,burnIn=1000)
#   yHatCV[tst]=fm$yHat[tst]
# }
# 
# proc.time()[3]-timeIn




# data(wheat)
# 
# nIter=12000
# burnIn=2000
# 
# X=scale(wheat.X)/sqrt(ncol(wheat.X))
# y=wheat.Y[,1]
# 
# fm1=BGLR( y=y,ETA=list(mrk=list(X=X,model='BRR')),
#           nIter=nIter,burnIn=burnIn,saveAt='brr_'
# )
# 
# varE=scan('brr_varE.dat')
# varU=scan('brr_ETA_mrk_varB.dat')
# h2_1=varU/(varU+varE)


# G=tcrossprod(X)
# fm2=BGLR( y=y,ETA=list(G=list(K=G,model='RKHS')),
#           nIter=nIter,burnIn=burnIn,saveAt='eig_'
# )
# varE=scan( 'eig_varE.dat')
# varU=scan('eig_ETA_G_varU.dat')
# h2_2=varU/(varU+varE)





