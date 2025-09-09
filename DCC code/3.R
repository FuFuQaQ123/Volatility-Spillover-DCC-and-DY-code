# 加载必要的包
library(rmgarch)
library(readxl)
library(openxlsx)
library(ggplot2)
library(reshape2)
library(moments)

# 1. 数据准备
# -----------------------------------------------------------
data <- read_excel("F:/课题组7/数据总表（3）对数化后.xlsx")
data <- data[, c("Date", "BDI", "BDTI", "BCTI", "GPR","WTI")]
data <- na.omit(data)


# 2. 定义滞后阶数测试范围
lags <- c(1, 2, 3, 4, 5, 10, 20)
# 3. 自动选择GARCH阶数函数
# -----------------------------------------------------------
auto_garch_order <- function(series, max_p = 1, max_q = 1) {
  best_aic <- Inf
  best_order <- c(1, 1)  # 默认使用GARCH(1,1)
  
  # 检查序列是否足够长
  if (length(series) < 50) {
    return(c(1, 1))
  }
  
  for (p in 0:max_p) {
    for (q in 0:max_q) {
      if (p == 0 & q == 0) next
      
      spec <- ugarchspec(
        variance.model = list(model = "sGARCH", garchOrder = c(p, q)),
        mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
        distribution.model = "norm"
      )
      
      fit <- tryCatch(
        {
          fit_result <- ugarchfit(spec, series, solver = "hybrid")
          if (fit_result@fit$convergence == 0) fit_result else NULL
        },
        error = function(e) NULL
      )
      
      if (!is.null(fit)) {
        current_aic <- infocriteria(fit)[1]
        if (current_aic < best_aic) {
          best_aic <- current_aic
          best_order <- c(p, q)
        }
      }
    }
  }
  return(best_order)
}

# 4. 显著性标记函数
# -----------------------------------------------------------
add_stars <- function(p) {
  ifelse(p < 0.01, "***",
         ifelse(p < 0.05, "**",
                ifelse(p < 0.1, "*", "")))
}

# 5. 处理相关系数的函数
# -----------------------------------------------------------
process_corr <- function(corr_array, i, j, dates) {
  corr_series <- sapply(1:dim(corr_array)[3], function(t) corr_array[i, j, t])
  
  # t检验
  t_test <- t.test(corr_series)
  
  return(list(
    series = corr_series,
    mean = mean(corr_series),
    p.value = t_test$p.value
  ))
}

# 6. 主分析流程 - 拟合所有滞后阶数
# -----------------------------------------------------------
results <- list()

# 记录开始时间
start_time <- Sys.time()
cat("开始时间:", format(start_time), "\n")

for (k in lags) {
  cat("\nProcessing lag", k, "...\n")
  
  # 生成滞后数据
  n <- nrow(data)
  lagged_GPR <- c(rep(NA, k), data$GPR[1:(n - k)])
  current_data <- cbind(data, GPR_lag = lagged_GPR)
  current_data <- na.omit(current_data)
  
  # 检查样本量是否足够
  if (nrow(current_data) < 100) {
    cat("警告: 滞后", k, "的样本量不足 (n =", nrow(current_data), "), 跳过此滞后阶数\n")
    next
  }
  
  dates <- current_data$Date
  
  # 提取变量矩阵
  y <- current_data[, c("BDI", "BDTI", "BCTI", "WTI", "GPR_lag")]
  colnames(y)[5] <- "GPR"
  
  # 对数据进行标准化，提高数值稳定性
  y_scaled <- as.data.frame(scale(y))
  
  # 使用固定的GARCH(1,1)模型
  specs <- list()
  for (i in 1:ncol(y_scaled)) {
    if (colnames(y_scaled)[i] == "BDTI") {
      specs[[i]] <- ugarchspec(
        variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
        mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
        distribution.model = "norm"
      )
    } else {
      specs[[i]] <- ugarchspec(
        variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
        mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
        distribution.model = "norm"
      )
    }
  }
  
  # 构建DCC模型
  mspec <- multispec(specs)
  dcc_spec <- dccspec(mspec, dccOrder = c(1, 1), model = "DCC")
  
  # 尝试多种方法拟合模型
  dcc_fit <- NULL
  
  # 方法1: 先拟合单变量GARCH，再拟合DCC
  tryCatch({
    multifit <- multifit(mspec, y_scaled)
    dcc_fit <- dccfit(dcc_spec, data = y_scaled, fit.control = list(eval.se = TRUE), fit = multifit)
    cat("Method 1 (multifit) succeeded for lag", k, "\n")
  }, error = function(e) {
    cat("Method 1 failed for lag", k, ":", e$message, "\n")
  })
  
  # 方法2: 如果方法1失败，尝试直接拟合
  if (is.null(dcc_fit)) {
    tryCatch({
      dcc_fit <- dccfit(dcc_spec, data = y_scaled, solver = "nlminb", 
                        fit.control = list(eval.se = TRUE))
      cat("Method 2 (direct fit) succeeded for lag", k, "\n")
    }, error = function(e) {
      cat("Method 2 failed for lag", k, ":", e$message, "\n")
    })
  }
  
  # 方法3: 如果方法2失败，尝试使用不同的优化器
  if (is.null(dcc_fit)) {
    tryCatch({
      dcc_fit <- dccfit(dcc_spec, data = y_scaled, solver = "solnp", 
                        fit.control = list(eval.se = TRUE))
      cat("Method 3 (solnp) succeeded for lag", k, "\n")
    }, error = function(e) {
      cat("Method 3 failed for lag", k, ":", e$message, "\n")
    })
  }
  
  # 方法4: 如果所有方法都失败，尝试简化模型（使用CCC而不是DCC）
  if (is.null(dcc_fit)) {
    tryCatch({
      ccc_spec <- dccspec(mspec, dccOrder = c(0, 0), model = "DCC")  # CCC模型
      dcc_fit <- dccfit(ccc_spec, data = y_scaled, solver = "solnp", 
                        fit.control = list(eval.se = TRUE))
      cat("Method 4 (CCC) succeeded for lag", k, "\n")
    }, error = function(e) {
      cat("Method 4 failed for lag", k, ":", e$message, "\n")
    })
  }
  
  if (!is.null(dcc_fit)) {
    # 提取参数矩阵
    matcoef <- dcc_fit@mfit$matcoef
    
    # 动态相关系数序列
    all_corrs <- rcor(dcc_fit)
    
    # 处理相关系数
    var_names <- colnames(y_scaled)
    pairs <- combn(var_names, 2, simplify = FALSE)
    corr_list <- list()
    
    for (pair in pairs) {
      i <- which(var_names == pair[1])
      j <- which(var_names == pair[2])
      corr_name <- paste0("Corr_", pair[1], "_", pair[2])
      corr_list[[corr_name]] <- process_corr(all_corrs, i, j, dates)
    }
    
    # 存储结果
    results[[paste0("lag_", k)]] <- list(
      corr_list = corr_list,
      std_resid = as.data.frame(dcc_fit@mfit$stdresid),
      all_corrs = all_corrs,
      dates = dates,
      y = y_scaled,
      alpha = list(
        coef = ifelse("[Joint]dcca1" %in% names(coef(dcc_fit)), 
                      coef(dcc_fit)["[Joint]dcca1"], NA),
        p.value = ifelse("[Joint]dcca1" %in% rownames(matcoef), 
                         matcoef["[Joint]dcca1", "Pr(>|t|)"], NA)
      ),
      beta = list(
        coef = ifelse("[Joint]dccb1" %in% names(coef(dcc_fit)), 
                      coef(dcc_fit)["[Joint]dccb1"], NA),
        p.value = ifelse("[Joint]dccb1" %in% rownames(matcoef), 
                         matcoef["[Joint]dccb1", "Pr(>|t|)"], NA)
      )
    )
    
    # 显示进度
    elapsed_time <- difftime(Sys.time(), start_time, units = "mins")
    cat("已完成滞后", k, "阶拟合，已用时:", round(elapsed_time, 2), "分钟\n")
  } else {
    cat("所有拟合方法均失败 for lag", k, "\n")
  }
}

# 记录结束时间
end_time <- Sys.time()
cat("\n所有滞后阶数拟合完成! 总用时:", round(difftime(end_time, start_time, units = "mins"), 2), "分钟\n")

# 7. 生成输出文件
# -----------------------------------------------------------

# 检查是否有成功拟合的结果
if (length(results) > 0) {
  # Table1: 标准化残差
  wb_std <- createWorkbook()
  addWorksheet(wb_std, "Standardized_Residuals")
  
  resid_table <- data.frame()
  for (k in lags) {
    lag_name <- paste0("lag_", k)
    if (!is.null(results[[lag_name]])) {
      res <- results[[lag_name]]
      for (col in colnames(res$std_resid)) {
        # 执行t检验
        t_test <- t.test(res$std_resid[[col]])
        # 格式化输出
        formatted <- sprintf("%.3f%s", 
                             mean(res$std_resid[[col]]),
                             add_stars(t_test$p.value))
        resid_table <- rbind(resid_table, data.frame(
          Lag = k,
          Variable = col,
          Mean = formatted,
          P_Value = sprintf("%.4f", t_test$p.value)
        ))
      }
    }
  }
  writeData(wb_std, "Standardized_Residuals", resid_table)
  saveWorkbook(wb_std, "C:/Users/Administrator/Desktop/1.xlsx", overwrite = TRUE)
  
  # Table2: 平均相关系数矩阵
  wb_corr <- createWorkbook()
  for (k in lags) {
    lag_name <- paste0("lag_", k)
    if (!is.null(results[[lag_name]])) {
      res <- results[[lag_name]]
      mean_corr <- apply(res$all_corrs, 1:2, mean)
      addWorksheet(wb_corr, paste0("Lag_", k))
      writeData(wb_corr, paste0("Lag_", k), as.data.frame(mean_corr), rowNames = TRUE)
    }
  }
  saveWorkbook(wb_corr, "C:/Users/Administrator/Desktop/2.xlsx", overwrite = TRUE)
  
  # 为每个滞后阶数创建动态相关系数时序图
  for (k in lags) {
    lag_name <- paste0("lag_", k)
    if (!is.null(results[[lag_name]])) {
      res <- results[[lag_name]]
      corr_array <- res$all_corrs
      dates <- res$dates
      var_names <- colnames(res$y)
      
      # 生成所有变量组合
      pairs <- combn(var_names, 2, simplify = FALSE)
      
      # 创建绘图数据
      plot_data <- data.frame()
      for (pair in pairs) {
        i <- which(var_names == pair[1])
        j <- which(var_names == pair[2])
        corr_series <- sapply(1:dim(corr_array)[3], function(t) corr_array[i, j, t])
        plot_data <- rbind(plot_data, data.frame(
          Date = rep(dates, length.out = length(corr_series)),
          Correlation = corr_series,
          Pair = paste(pair[1], pair[2], sep = "_")
        ))
      }
      
      # 绘制图表
      p <- ggplot(plot_data, aes(x = Date, y = Correlation, color = Pair)) +
        geom_line() +
        facet_wrap(~Pair, ncol = 2, scales = "free_y") +
        labs(title = paste("Dynamic Conditional Correlations (Lag =", k, ")")) +
        theme_bw() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1),
              legend.position = "none")
      
      ggsave(paste0("C:/Users/Administrator/Desktop/dcc_plot_lag_", k, ".png"), 
             plot = p, width = 12, height = 10, dpi = 300)
    }
  }
  
  # Table3: 相关系数描述统计
  wb_desc <- createWorkbook()
  for (k in lags) {
    lag_name <- paste0("lag_", k)
    if (!is.null(results[[lag_name]])) {
      res <- results[[lag_name]]
      corr_array <- res$all_corrs
      var_names <- colnames(res$y)
      
      # 计算描述统计
      desc_stats <- data.frame()
      for (i in 1:(length(var_names)-1)) {
        for (j in (i+1):length(var_names)) {
          series <- sapply(1:dim(corr_array)[3], function(t) corr_array[i, j, t])
          desc <- data.frame(
            Lag = k,
            Pair = paste(var_names[i], var_names[j], sep = "_"),
            Mean = mean(series),
            SD = sd(series),
            Min = min(series),
            Max = max(series),
            Skewness = skewness(series),
            Kurtosis = kurtosis(series)
          )
          desc_stats <- rbind(desc_stats, desc)
        }
      }
      
      addWorksheet(wb_desc, paste0("Lag_", k))
      writeData(wb_desc, paste0("Lag_", k), desc_stats)
    }
  }
  saveWorkbook(wb_desc, "C:/Users/Administrator/Desktop/3.xlsx", overwrite = TRUE)
  
  # Table4: 滞后相关系数检验
  wb_lag <- createWorkbook()
  addWorksheet(wb_lag, "Results")
  
  final_table <- data.frame()
  for (k in lags) {
    lag_name <- paste0("lag_", k)
    if (!is.null(results[[lag_name]])) {
      res <- results[[lag_name]]
      var_names <- colnames(res$y)
      pairs <- combn(var_names, 2, simplify = FALSE)
      
      for (pair in pairs) {
        corr_name <- paste0("Corr_", pair[1], "_", pair[2])
        if (!is.null(res$corr_list[[corr_name]])) {
          # 合并均值和显著性标记
          formatted_mean <- sprintf("%.3f%s", 
                                    res$corr_list[[corr_name]]$mean,
                                    add_stars(res$corr_list[[corr_name]]$p.value))
          
          row <- data.frame(
            Lag = k,
            Pair = paste(pair[1], pair[2], sep = " vs "),
            Mean = formatted_mean,
            P_Value = sprintf("%.4f", res$corr_list[[corr_name]]$p.value)
          )
          final_table <- rbind(final_table, row)
        }
      }
    }
  }
  
  if (nrow(final_table) > 0) {
    writeData(wb_lag, "Results", final_table)
    saveWorkbook(wb_lag, "C:/Users/Administrator/Desktop/4.xlsx", overwrite = TRUE)
  }
  
  # Table5: DCC参数输出
  wb_dcc <- createWorkbook()
  addWorksheet(wb_dcc, "DCC_Parameters")
  
  dcc_params <- data.frame()
  for (k in lags) {
    lag_name <- paste0("lag_", k)
    if (!is.null(results[[lag_name]])) {
      res <- results[[lag_name]]
      
      # 检查参数是否存在
      if (!is.null(res$alpha$coef) && !is.na(res$alpha$coef)) {
        alpha_str <- sprintf("%.3f%s", 
                             res$alpha$coef,
                             add_stars(res$alpha$p.value))
      } else {
        alpha_str <- "NA"
      }
      
      if (!is.null(res$beta$coef) && !is.na(res$beta$coef)) {
        beta_str <- sprintf("%.3f%s", 
                            res$beta$coef,
                            add_stars(res$beta$p.value))
      } else {
        beta_str <- "NA"
      }
      
      dcc_params <- rbind(dcc_params, data.frame(
        Lag = k,
        Parameter = c("alpha", "beta"),
        Estimate = c(alpha_str, beta_str),
        P_Value = c(ifelse(!is.null(res$alpha$p.value), sprintf("%.4f", res$alpha$p.value), "NA"),
                    ifelse(!is.null(res$beta$p.value), sprintf("%.4f", res$beta$p.value), "NA"))
      ))
    }
  }
  writeData(wb_dcc, "DCC_Parameters", dcc_params)
  saveWorkbook(wb_dcc, "C:/Users/Administrator/Desktop/5.xlsx", overwrite = TRUE)
  
  cat("所有输出文件已生成!\n")
} else {
  cat("没有成功拟合的模型。请检查数据和模型设定。\n")
}