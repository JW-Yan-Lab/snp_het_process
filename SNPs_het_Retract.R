library(vcfR)
folder_search = commandArgs(trailingONly=TRUE)
folder_search = folder_search[1]
files = list.files(folder_search)


v_temp = read.vcfR(files[1])
gt_temp = data.frame(v_temp@gt)
gt_temp$FORMAT = NULL
subj = colnames(gt_temp)
for(i in 1:length(subj)){subj[i] = strsplit(subj[i],'X')[[1]][2]}
colnames(gt_temp) = subj
fix_temp = data.frame(v_temp@fix)
fix_temp$INFO = NULL
fix_temp$NAME = paste0(fix_temp$CHROM,'__',fix_temp$POS)
rownames(gt_temp) = fix_temp$NAME
v_main = gt_temp

for(f in 2:length(files)){
	v_temp = read.vcfR(files[f])
	gt_temp = data.frame(v_temp@gt)
	gt_temp$FORMAT = NULL
	subj = colnames(gt_temp)
	for(i in 1:length(subj)){subj[i] = strsplit(subj[i],'X')[[1]][2]}
	colnames(gt_temp) = subj
	fix_temp = data.frame(v_temp@fix)
	fix_temp$INFO = NULL
	fix_temp$NAME = paste0(fix_temp$CHROM,'__',fix_temp$POS)
	rownames(gt_temp) = fix_temp$NAME
	v_main = rbind(v_main,gt_temp)
}

final_matrix = matrix('a',
               nrow = nrow(v_main),
               ncol = ncol(v_main))

row.names(final_matrix) = row.names(v_main)
colnames(final_matrix) = colnames(v_main)

for(i in 1:nrow(v_main)){
	for(j in 1:ncol(v_main)){
		 final_matrix[i,j] = strsplit(as.character(v_main[i,j]),'[:]')[[1]][1]
	}
}

write.csv(final_matrix,"Het_Info_MSBB_820_SNPs.csv",quote = FALSE)
