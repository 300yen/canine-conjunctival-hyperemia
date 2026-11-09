# Clustered Wilcoxon rank-sum comparisons, Rosner-Glynn-Lee (RGL) method.
# Run: Rscript compare_disease_groups.R features.csv outputs
# Requires clusrank 1.0-4. Outputs contain aggregate statistics only.

compare_disease_groups <- function(data) {
  required <- c('ID', 'R/L', 'Category_Mid', 'A_label', 'B_label', 'C_label',
                'AndMask_LabA_Density', 'fractal_dimension')
  if (!all(required %in% names(data))) stop('Missing required input columns')
  if (anyNA(data[c('ID', 'R/L', 'Category_Mid')])) stop('Missing eye key or diagnosis')
  if (anyDuplicated(data[c('ID', 'R/L')])) stop('Duplicate dog/eye keys')
  ratings <- as.matrix(data[c('A_label', 'B_label', 'C_label')])
  if (anyNA(ratings) || !all(ratings %in% c(0:3, 5))) stop('Invalid expert ratings')
  data <- data[rowSums(ratings == 5) == 0, , drop = FALSE]
  data$expert_mean <- rowMeans(data[c('A_label', 'B_label', 'C_label')])
  mapping <- c(
    '00_Normal' = 'Normal control', '13_ISP' = 'Intrascleral prosthesis',
    '01_Cornea_Acute_Ulcer' = 'Corneal ulcer',
    '02_Cornea_KCS_Exposure' = 'Non-ulcerative corneo-conjunctival disease',
    '03_Cornea_Chronic_Others' = 'Non-ulcerative corneo-conjunctival disease',
    '04_Scleritis' = 'Scleritis', '05_Lens_Ant_Luxation' = 'Anterior lens luxation',
    '07_Lens_Other_Luxation' = 'Posterior/other lens luxation',
    '06_Lens_Cataract' = 'Cataract', '08_Uveitis' = 'Uveitis (including LIU)',
    '09_Glaucoma' = 'Glaucoma',
    '10_Retina_Neuro_NonUrgent' = 'Retinal/neurological (non-neuritis)',
    '14_Optic_Neuritis_Urgent' = 'Optic neuritis',
    '11_Tumor_Intraocular' = 'Intraocular tumor',
    '12_Tumor_Extra_Adnexa' = 'Extraocular/adnexal tumor')
  data$disease_group <- unname(mapping[data$Category_Mid])
  if (anyNA(data$disease_group)) stop('Unmapped disease category')
  targets <- sort(setdiff(unique(mapping), 'Normal control'))
  outcomes <- c(expert_mean = 'Mean expert grade', AndMask_LabA_Density = 'RS',
                fractal_dimension = 'FD')
  rows <- list()
  for (variable in names(outcomes)) {
    eligible <- data[!is.na(data[[variable]]), , drop = FALSE]
    if (!all(is.finite(eligible[[variable]]))) stop('Non-finite outcome')
    for (target in targets) {
      pair <- eligible[eligible$disease_group %in% c('Normal control', target), , drop = FALSE]
      sizes <- table(as.character(pair$ID))
      if (any(sizes > 2)) stop('More than two eligible eyes per dog')
      # clusrank 1.0-4 unbalanced subunit RGL assembles scores by cluster size.
      # Assign temporary cluster numbers in that order to preserve alignment.
      dogs <- names(sizes)[order(as.integer(sizes), names(sizes))]
      pair$cluster <- match(as.character(pair$ID), dogs)
      pair <- pair[order(pair$cluster, pair[['R/L']]), , drop = FALSE]
      pair$group <- ifelse(pair$disease_group == target, 1, 2)
      disease <- pair[pair$group == 1, , drop = FALSE]
      control <- pair[pair$group == 2, , drop = FALSE]
      if (!nrow(disease) || !nrow(control)) stop('Missing group in the 13-comparison family')
      fit <- clusrank::clusWilcox.test(pair[[variable]], cluster = pair$cluster,
        group = pair$group, method = 'rgl', exact = FALSE,
        alternative = 'two.sided', mu = 0)
      if (!is.finite(fit$p.value)) stop('Non-finite RGL p value')
      cq <- quantile(control[[variable]], c(.25, .5, .75), names = FALSE)
      dq <- quantile(disease[[variable]], c(.25, .5, .75), names = FALSE)
      rows[[length(rows) + 1L]] <- data.frame(
        outcome = outcomes[[variable]], disease_group = target,
        control_n_eyes = nrow(control), control_n_dogs = length(unique(control$cluster)),
        disease_n_eyes = nrow(disease), disease_n_dogs = length(unique(disease$cluster)),
        control_median = cq[2], control_q1 = cq[1], control_q3 = cq[3],
        disease_median = dq[2], disease_q1 = dq[1], disease_q3 = dq[3],
        median_difference = dq[2] - cq[2],
        # The package recodes group order internally; report |Z| for this two-sided test.
        rgl_abs_z = abs(as.numeric(fit$statistic)), rgl_p = fit$p.value,
        descriptive_only = target == 'Intrascleral prosthesis')
    }
  }
  result <- do.call(rbind, rows)
  result$holm_adjusted_p <- ave(result$rgl_p, result$outcome,
    FUN = function(p) p.adjust(p, method = 'holm'))
  result$significant_0_05 <- result$holm_adjusted_p < .05
  result
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  if (length(args) != 2L) stop('Usage: Rscript compare_disease_groups.R features.csv output_directory')
  if (!requireNamespace('clusrank', quietly = TRUE)) stop('Install clusrank 1.0-4 first')
  if (as.character(utils::packageVersion('clusrank')) != '1.0.4') {
    stop('This analysis was verified with clusrank 1.0-4; install that version')
  }
  data <- read.csv(args[1], check.names = FALSE, stringsAsFactors = FALSE,
                   fileEncoding = 'UTF-8-BOM')
  result <- compare_disease_groups(data)
  dir.create(args[2], recursive = TRUE, showWarnings = FALSE)
  write.csv(result, file.path(args[2], 'control_vs_disease_rgl.csv'), row.names = FALSE)
  capture.output(sessionInfo(), file = file.path(args[2], 'disease_comparison_session.txt'))
}

if (sys.nframe() == 0L) main()
