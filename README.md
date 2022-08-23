# snp_het_process

##Step 1: Using VCF tools to extract vcf files with the snp id or chr/pos information
Use --snps if using snp ID denoting a file with snp id in each row 
Use --positions if using chromosome and base pair position 

Example for iteration over all .vcf.gz files
```
for file in *.gz;
do vcftools --gzvcf $file --positions ~/Position_822_MSBB.txt --recode --recode-INFO-all --out MSBB_822_$file;
done
```
Example for iteration over all .vcf files

for file in *.gz;
do vcftools --vcf $file --positions ~/Position_822_MSBB.txt --recode --recode-INFO-all --out MSBB_822_$file;
done


##Step 2: Use Code block provided with vcfR installed (in R)

```R
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
```
To use the code, install the vcfR package and run as

```
Rscript SNPs_Het_Retract.R folder_name
```

where the folder name consists of all the vcf files obtained from the vcftools


## Result Outlook
Here the rownames are in CHR__POS format seperated by two underscores and the colnames are in subject IDs.
The information inside would be for both alleles where 0,1,2 are the values with  
	0 for the reference allele, 
	1 for the first allele listed in ALT column,
	2 for the second allele listed in ALT. 




	
	
