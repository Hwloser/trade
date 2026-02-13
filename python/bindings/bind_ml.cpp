#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/eigen/dense.h>

#ifdef HAVE_LIGHTGBM
#include "trade/ml/lgbm_model.h"
#include "trade/ml/model_trainer.h"
#include "trade/ml/model_evaluator.h"
#endif

namespace nb = nanobind;

void bind_ml(nb::module_& m) {
#ifdef HAVE_LIGHTGBM
    auto ml = m.def_submodule("ml", "Machine learning models");

    // ========================================================================
    // LGBMParams
    // ========================================================================
    nb::class_<trade::LGBMParams>(ml, "LGBMParams")
        .def(nb::init<>())
        .def_rw("objective", &trade::LGBMParams::objective)
        .def_rw("metric", &trade::LGBMParams::metric)
        .def_rw("num_leaves", &trade::LGBMParams::num_leaves)
        .def_rw("max_depth", &trade::LGBMParams::max_depth)
        .def_rw("learning_rate", &trade::LGBMParams::learning_rate)
        .def_rw("n_estimators", &trade::LGBMParams::n_estimators)
        .def_rw("feature_fraction", &trade::LGBMParams::feature_fraction)
        .def_rw("bagging_fraction", &trade::LGBMParams::bagging_fraction)
        .def_rw("bagging_freq", &trade::LGBMParams::bagging_freq)
        .def_rw("lambda_l1", &trade::LGBMParams::lambda_l1)
        .def_rw("lambda_l2", &trade::LGBMParams::lambda_l2)
        .def_rw("min_data_in_leaf", &trade::LGBMParams::min_data_in_leaf)
        .def_rw("num_threads", &trade::LGBMParams::num_threads)
        .def_rw("verbose", &trade::LGBMParams::verbose)
        .def_rw("early_stopping_rounds", &trade::LGBMParams::early_stopping_rounds)
        .def_rw("seed", &trade::LGBMParams::seed)
        .def("to_param_string", &trade::LGBMParams::to_param_string)
        .def_static("from_map", &trade::LGBMParams::from_map,
                     nb::arg("param_map"),
                     "Build LGBMParams from a key-value string map");

    // ========================================================================
    // LGBMTrainResult
    // ========================================================================
    nb::class_<trade::LGBMTrainResult>(ml, "LGBMTrainResult")
        .def(nb::init<>())
        .def_rw("best_iteration", &trade::LGBMTrainResult::best_iteration)
        .def_rw("best_score", &trade::LGBMTrainResult::best_score)
        .def_rw("metric_name", &trade::LGBMTrainResult::metric_name)
        .def_rw("n_features", &trade::LGBMTrainResult::n_features)
        .def_rw("n_train_samples", &trade::LGBMTrainResult::n_train_samples)
        .def_rw("n_valid_samples", &trade::LGBMTrainResult::n_valid_samples)
        .def_rw("train_time_seconds", &trade::LGBMTrainResult::train_time_seconds);

    // ========================================================================
    // LGBMModel
    // ========================================================================
    nb::class_<trade::LGBMModel>(ml, "LGBMModel")
        .def(nb::init<>())
        .def("train", &trade::LGBMModel::train,
             nb::arg("features"), nb::arg("labels"), nb::arg("params"),
             nb::arg("valid_features") = Eigen::MatrixXd(),
             nb::arg("valid_labels") = Eigen::VectorXd())
        .def("predict", &trade::LGBMModel::predict)
        .def("predict_one", &trade::LGBMModel::predict_one)
        .def("feature_importance", &trade::LGBMModel::feature_importance,
             nb::arg("importance_type") = 1)
        .def("feature_importance_named", &trade::LGBMModel::feature_importance_named,
             nb::arg("feature_names"), nb::arg("importance_type") = 1)
        .def("shap_values", &trade::LGBMModel::shap_values)
        .def("save", &trade::LGBMModel::save)
        .def("load", &trade::LGBMModel::load)
        .def("to_string", &trade::LGBMModel::to_string)
        .def("from_string", &trade::LGBMModel::from_string)
        .def("is_trained", &trade::LGBMModel::is_trained)
        .def("num_features", &trade::LGBMModel::num_features)
        .def("num_iterations", &trade::LGBMModel::num_iterations)
        .def("num_classes", &trade::LGBMModel::num_classes);

    // ========================================================================
    // SplitIndices
    // ========================================================================
    nb::class_<trade::SplitIndices>(ml, "SplitIndices")
        .def(nb::init<>())
        .def_rw("train", &trade::SplitIndices::train)
        .def_rw("valid", &trade::SplitIndices::valid)
        .def_rw("test", &trade::SplitIndices::test);

    // ========================================================================
    // FoldSpec
    // ========================================================================
    nb::class_<trade::FoldSpec>(ml, "FoldSpec")
        .def(nb::init<>())
        .def_rw("fold_id", &trade::FoldSpec::fold_id)
        .def_rw("indices", &trade::FoldSpec::indices);

    // ========================================================================
    // FoldResult
    // ========================================================================
    nb::class_<trade::FoldResult>(ml, "FoldResult")
        .def(nb::init<>())
        .def_rw("fold_id", &trade::FoldResult::fold_id)
        .def_rw("train_result", &trade::FoldResult::train_result)
        .def_rw("test_ic", &trade::FoldResult::test_ic)
        .def_rw("test_rank_ic", &trade::FoldResult::test_rank_ic)
        .def_rw("test_mse", &trade::FoldResult::test_mse)
        .def_rw("test_mae", &trade::FoldResult::test_mae)
        .def_rw("predictions", &trade::FoldResult::predictions)
        .def_rw("actuals", &trade::FoldResult::actuals);

    // ========================================================================
    // TrainingPipelineResult
    // ========================================================================
    nb::class_<trade::TrainingPipelineResult>(ml, "TrainingPipelineResult")
        .def(nb::init<>())
        .def_rw("fold_results", &trade::TrainingPipelineResult::fold_results)
        .def_rw("mean_test_ic", &trade::TrainingPipelineResult::mean_test_ic)
        .def_rw("mean_test_rank_ic", &trade::TrainingPipelineResult::mean_test_rank_ic)
        .def_rw("std_test_ic", &trade::TrainingPipelineResult::std_test_ic)
        .def_rw("best_model_path", &trade::TrainingPipelineResult::best_model_path)
        .def_rw("best_fold_id", &trade::TrainingPipelineResult::best_fold_id)
        .def_rw("best_params", &trade::TrainingPipelineResult::best_params);

    // ========================================================================
    // ModelTrainer::Config
    // ========================================================================
    nb::class_<trade::ModelTrainer::Config>(ml, "ModelTrainerConfig")
        .def(nb::init<>())
        .def_rw("train_years", &trade::ModelTrainer::Config::train_years)
        .def_rw("test_years", &trade::ModelTrainer::Config::test_years)
        .def_rw("step_years", &trade::ModelTrainer::Config::step_years)
        .def_rw("n_folds", &trade::ModelTrainer::Config::n_folds)
        .def_rw("prediction_horizon", &trade::ModelTrainer::Config::prediction_horizon)
        .def_rw("purge_gap", &trade::ModelTrainer::Config::purge_gap)
        .def_rw("embargo_days", &trade::ModelTrainer::Config::embargo_days)
        .def_rw("model_output_dir", &trade::ModelTrainer::Config::model_output_dir)
        .def_rw("model_name_prefix", &trade::ModelTrainer::Config::model_name_prefix)
        .def_rw("feature_names", &trade::ModelTrainer::Config::feature_names)
        .def_rw("label_name", &trade::ModelTrainer::Config::label_name)
        .def("resolve_defaults", &trade::ModelTrainer::Config::resolve_defaults,
             nb::arg("train_size"),
             "Resolve default values for purge_gap and embargo_days");

    // ========================================================================
    // ModelTrainer
    // ========================================================================
    nb::class_<trade::ModelTrainer>(ml, "ModelTrainer")
        .def(nb::init<trade::ModelTrainer::Config>())
        .def("time_series_split", &trade::ModelTrainer::time_series_split,
             nb::arg("n_samples"), nb::arg("train_ratio") = 0.8,
             "Simple time-series split (single train/test cut)")
        .def("run_pipeline", &trade::ModelTrainer::run_pipeline,
             nb::arg("features"), nb::arg("labels"), nb::arg("dates"),
             nb::arg("params"), nb::arg("use_kfold") = false,
             "Full pipeline: split -> train -> evaluate -> save best model")
        .def("train_fold", &trade::ModelTrainer::train_fold,
             nb::arg("features"), nb::arg("labels"),
             nb::arg("fold"), nb::arg("params"),
             "Train and evaluate on a single fold")
        .def_static("slice_rows",
             nb::overload_cast<const Eigen::MatrixXd&, const std::vector<int>&>(
                 &trade::ModelTrainer::slice_rows),
             nb::arg("mat"), nb::arg("indices"),
             "Extract sub-matrix by row indices")
        .def_static("slice_rows_vec",
             nb::overload_cast<const Eigen::VectorXd&, const std::vector<int>&>(
                 &trade::ModelTrainer::slice_rows),
             nb::arg("vec"), nb::arg("indices"),
             "Extract sub-vector by row indices")
        .def("config", &trade::ModelTrainer::config, nb::rv_policy::reference_internal);

    // ========================================================================
    // ModelEvaluator result structs
    // ========================================================================

    // HorizonICResult
    nb::class_<trade::HorizonICResult>(ml, "HorizonICResult")
        .def(nb::init<>())
        .def_rw("horizon", &trade::HorizonICResult::horizon)
        .def_rw("ic", &trade::HorizonICResult::ic)
        .def_rw("rank_ic", &trade::HorizonICResult::rank_ic)
        .def_rw("ic_ir", &trade::HorizonICResult::ic_ir)
        .def_rw("rank_ic_ir", &trade::HorizonICResult::rank_ic_ir)
        .def_rw("n_periods", &trade::HorizonICResult::n_periods);

    // FeatureImportanceEntry
    nb::class_<trade::FeatureImportanceEntry>(ml, "FeatureImportanceEntry")
        .def(nb::init<>())
        .def_rw("name", &trade::FeatureImportanceEntry::name)
        .def_rw("gain_importance", &trade::FeatureImportanceEntry::gain_importance)
        .def_rw("split_importance", &trade::FeatureImportanceEntry::split_importance)
        .def_rw("shap_mean_abs", &trade::FeatureImportanceEntry::shap_mean_abs)
        .def_rw("rank_gain", &trade::FeatureImportanceEntry::rank_gain)
        .def_rw("rank_shap", &trade::FeatureImportanceEntry::rank_shap);

    // CalibrationBin
    nb::class_<trade::CalibrationBin>(ml, "CalibrationBin")
        .def(nb::init<>())
        .def_rw("pred_low", &trade::CalibrationBin::pred_low)
        .def_rw("pred_high", &trade::CalibrationBin::pred_high)
        .def_rw("pred_mean", &trade::CalibrationBin::pred_mean)
        .def_rw("actual_freq", &trade::CalibrationBin::actual_freq)
        .def_rw("count", &trade::CalibrationBin::count);

    // DSRResult
    nb::class_<trade::DSRResult>(ml, "DSRResult")
        .def(nb::init<>())
        .def_rw("observed_sharpe", &trade::DSRResult::observed_sharpe)
        .def_rw("dsr", &trade::DSRResult::dsr)
        .def_rw("dsr_pvalue", &trade::DSRResult::dsr_pvalue)
        .def_rw("n_trials", &trade::DSRResult::n_trials)
        .def_rw("expected_max_sharpe", &trade::DSRResult::expected_max_sharpe);

    // PBOResult
    nb::class_<trade::PBOResult>(ml, "PBOResult")
        .def(nb::init<>())
        .def_rw("pbo", &trade::PBOResult::pbo)
        .def_rw("n_combinations", &trade::PBOResult::n_combinations)
        .def_rw("n_overfit", &trade::PBOResult::n_overfit)
        .def_rw("logit_mean", &trade::PBOResult::logit_mean)
        .def_rw("logit_std", &trade::PBOResult::logit_std);

    // BootstrapSharpeCI
    nb::class_<trade::BootstrapSharpeCI>(ml, "BootstrapSharpeCI")
        .def(nb::init<>())
        .def_rw("point_estimate", &trade::BootstrapSharpeCI::point_estimate)
        .def_rw("ci_lower", &trade::BootstrapSharpeCI::ci_lower)
        .def_rw("ci_upper", &trade::BootstrapSharpeCI::ci_upper)
        .def_rw("confidence_level", &trade::BootstrapSharpeCI::confidence_level)
        .def_rw("n_bootstrap", &trade::BootstrapSharpeCI::n_bootstrap);

    // FDRResult::Entry
    nb::class_<trade::FDRResult::Entry>(ml, "FDREntry")
        .def(nb::init<>())
        .def_rw("index", &trade::FDRResult::Entry::index)
        .def_rw("p_value", &trade::FDRResult::Entry::p_value)
        .def_rw("adjusted_p", &trade::FDRResult::Entry::adjusted_p)
        .def_rw("significant", &trade::FDRResult::Entry::significant);

    // FDRResult
    nb::class_<trade::FDRResult>(ml, "FDRResult")
        .def(nb::init<>())
        .def_rw("alpha", &trade::FDRResult::alpha)
        .def_rw("total_tests", &trade::FDRResult::total_tests)
        .def_rw("significant_count", &trade::FDRResult::significant_count)
        .def_rw("entries", &trade::FDRResult::entries);

    // EvaluationReport
    nb::class_<trade::EvaluationReport>(ml, "EvaluationReport")
        .def(nb::init<>())
        .def_rw("ic_results", &trade::EvaluationReport::ic_results)
        .def_rw("feature_importance", &trade::EvaluationReport::feature_importance)
        .def_rw("calibration", &trade::EvaluationReport::calibration)
        .def_rw("dsr", &trade::EvaluationReport::dsr)
        .def_rw("pbo", &trade::EvaluationReport::pbo)
        .def_rw("sharpe_ci", &trade::EvaluationReport::sharpe_ci)
        .def_rw("fdr", &trade::EvaluationReport::fdr)
        .def_rw("overall_rank_ic", &trade::EvaluationReport::overall_rank_ic)
        .def_rw("overall_ic_ir", &trade::EvaluationReport::overall_ic_ir)
        .def_rw("passes_dsr_test", &trade::EvaluationReport::passes_dsr_test)
        .def_rw("passes_pbo_test", &trade::EvaluationReport::passes_pbo_test)
        .def_rw("sharpe_ci_excludes_zero", &trade::EvaluationReport::sharpe_ci_excludes_zero);

    // ========================================================================
    // ModelEvaluator (all static methods)
    // ========================================================================
    nb::class_<trade::ModelEvaluator>(ml, "ModelEvaluator")
        // IC evaluation
        .def_static("evaluate_ic", &trade::ModelEvaluator::evaluate_ic,
                     nb::arg("predictions"), nb::arg("price_panel"),
                     nb::arg("horizons"),
                     "Compute IC and Rank IC across multiple forward horizons")
        .def_static("evaluate_ic_single", &trade::ModelEvaluator::evaluate_ic_single,
                     nb::arg("predictions"), nb::arg("forward_returns"),
                     nb::arg("horizon"),
                     "Evaluate IC on a single horizon from pre-computed returns")
        // Feature importance ranking
        .def_static("rank_features", &trade::ModelEvaluator::rank_features,
                     nb::arg("gain_importance"), nb::arg("split_importance"),
                     nb::arg("shap_matrix"), nb::arg("feature_names"),
                     "Rank features by gain, split, and mean |SHAP|")
        // SHAP analysis
        .def_static("mean_abs_shap", &trade::ModelEvaluator::mean_abs_shap,
                     nb::arg("shap_matrix"),
                     "Compute per-feature mean absolute SHAP value")
        .def_static("top_k_shap", &trade::ModelEvaluator::top_k_shap,
                     nb::arg("shap_row"), nb::arg("feature_names"),
                     nb::arg("k") = 10,
                     "Top-K most impactful features for a single prediction")
        // Calibration
        .def_static("calibration_curve", &trade::ModelEvaluator::calibration_curve,
                     nb::arg("predicted"), nb::arg("actual"),
                     nb::arg("n_bins") = 10,
                     "Compute calibration bins for predicted vs actual")
        .def_static("brier_score", &trade::ModelEvaluator::brier_score,
                     nb::arg("predicted"), nb::arg("actual"),
                     "Brier score = mean((predicted - actual)^2)")
        // Deflated Sharpe Ratio
        .def_static("deflated_sharpe_ratio", &trade::ModelEvaluator::deflated_sharpe_ratio,
                     nb::arg("returns"), nb::arg("n_trials"),
                     "Compute the Deflated Sharpe Ratio")
        .def_static("expected_max_sharpe", &trade::ModelEvaluator::expected_max_sharpe,
                     nb::arg("n_trials"),
                     "Expected maximum Sharpe under the null")
        // Probability of Backtest Overfitting
        .def_static("probability_of_backtest_overfitting",
                     &trade::ModelEvaluator::probability_of_backtest_overfitting,
                     nb::arg("strategy_returns"),
                     nb::arg("n_partitions") = 16,
                     "Compute PBO using CSCV framework")
        // Benjamini-Hochberg FDR
        .def_static("benjamini_hochberg", &trade::ModelEvaluator::benjamini_hochberg,
                     nb::arg("p_values"), nb::arg("alpha") = 0.05,
                     "Apply Benjamini-Hochberg False Discovery Rate correction")
        // Bootstrap Sharpe CI
        .def_static("bootstrap_sharpe_ci", &trade::ModelEvaluator::bootstrap_sharpe_ci,
                     nb::arg("returns"),
                     nb::arg("n_bootstrap") = 10000,
                     nb::arg("confidence") = 0.95,
                     nb::arg("seed") = 42,
                     "Bootstrap confidence interval for the Sharpe ratio")
        // Full evaluation report
        .def_static("full_evaluation", &trade::ModelEvaluator::full_evaluation,
                     nb::arg("predictions"), nb::arg("price_panel"),
                     nb::arg("strategy_returns"),
                     nb::arg("gain_importance"), nb::arg("split_importance"),
                     nb::arg("shap_matrix"), nb::arg("feature_names"),
                     nb::arg("horizons") = std::vector<int>{1, 2, 5, 10, 20},
                     nb::arg("n_trials") = 1,
                     nb::arg("strategy_returns_matrix") = Eigen::MatrixXd(),
                     "Run all evaluations and produce a comprehensive report")
        // Utilities
        .def_static("sharpe_ratio", &trade::ModelEvaluator::sharpe_ratio,
                     nb::arg("returns"),
                     nb::arg("trading_days_per_year") = 242,
                     "Compute annualised Sharpe ratio from daily returns")
        .def_static("skewness", &trade::ModelEvaluator::skewness,
                     nb::arg("data"),
                     "Compute sample skewness")
        .def_static("kurtosis", &trade::ModelEvaluator::kurtosis,
                     nb::arg("data"),
                     "Compute excess kurtosis");
#endif // HAVE_LIGHTGBM
}
