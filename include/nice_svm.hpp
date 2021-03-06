//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
// http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef NICE_SVM_HPP
#define NICE_SVM_HPP

#include <random>
#include <utility>
#include <algorithm>
#include <limits>

#include "cpp_utils/algorithm.hpp"

#include "svm.h"

namespace svm {

/*!
 * \brief Describe a problem for SVM
 */
struct problem {
    std::size_t n_samples;  ///< The number of samples
    std::size_t n_features; ///< THe number of features per sample
    svm_problem sub;        ///< The real libsvm problem

    /*!
     * \brief Create a new empty problem
     */
    problem()
            : n_samples(0), n_features(0), sub({0, nullptr, nullptr}) {
        //Nothing to init
    }

    problem(std::size_t n_samples, std::size_t n_features)
            : n_samples(n_samples),
              n_features(n_features),
              sub({static_cast<int>(n_samples), new double[n_samples], new svm_node*[n_samples]}) {
        //Init the sub vectors
        for (std::size_t s = 0; s < n_samples; ++s) {
            sub.x[s] = new svm_node[n_features + 1];
        }
    }

    problem(problem& rhs) = delete;

    problem(problem&& rhs)
            : n_samples(rhs.n_samples),
              n_features(rhs.n_features),
              sub({rhs.sub.l, rhs.sub.y, rhs.sub.x}) {
        rhs.sub.x = nullptr;
        rhs.sub.y = nullptr;
    }

    problem& operator=(problem&& rhs) {
        n_samples    = rhs.n_samples;
        n_features   = rhs.n_features;
        sub.l        = rhs.sub.l;
        sub.x        = rhs.sub.x;
        sub.y        = rhs.sub.y;

        rhs.sub.x = nullptr;
        rhs.sub.y = nullptr;

        return *this;
    }

    ~problem() {
        if (sub.x) {
            for (std::size_t s = 0; s < n_samples; ++s) {
                delete[] sub.x[s];
            }

            delete[] sub.x;
        }

        if (sub.y) {
            delete[] sub.y;
        }
    }

    void scale() {
        double a = 0.0;
        double b = 1.0;

        for (std::size_t i = 0; i < n_features; ++i) {
            double min = std::numeric_limits<double>::max();
            double max = std::numeric_limits<double>::lowest();
            for (std::size_t n = 0; n < n_samples; ++n) {
                min = std::min(min, sample(n)[i].value);
                max = std::max(max, sample(n)[i].value);
            }

            for (std::size_t n = 0; n < n_samples; ++n) {
                sample(n)[i].value = a + ((b - a) * (sample(n)[i].value - min)) / (max - min);
            }
        }
    }

    svm_problem& get_problem() {
        return sub;
    }

    const svm_problem& get_problem() const {
        return sub;
    }

    double& label(std::size_t i) {
        return sub.y[i];
    }

    svm_node* sample(std::size_t i) {
        return sub.x[i];
    }
};

/*!
 * \brief Represent a trained model of SVM
 */
struct model {
    svm_model* sub = nullptr;

    /*!
     * \brief Construct a default model
     */
    model() = default;

    model(svm_model* sub)
            : sub(sub) {}

    model(model&& rhs)
            : sub(rhs.sub) {
        rhs.sub = nullptr;
    }

    model& operator=(model&& rhs) {
        sub        = rhs.sub;
        rhs.sub    = nullptr;

        return *this;
    }

    /*!
     * \brief Destroys the model
     */
    ~model() {
        if (sub) {
            svm_free_and_destroy_model(&sub);
        }
    }

    /*!
     * \brief Returns the number of classes of hte model
     * \return the number of classes of the model
     */
    std::size_t classes() {
        return svm_get_nr_class(sub);
    }

    /*!
     * \brief Returns a pointer to the libsvm model
     * \return a pointer to the libsvm model
     */
    svm_model* get_model() {
        return sub;
    }

    /*!
     * \brief Returns a pointer to the libsvm model
     * \return a pointer to the libsvm model
     */
    const svm_model* get_model() const {
        return sub;
    }

    /*!
     * \brief Indicates if the model is valid or not
     */
    operator bool() {
        return sub;
    }
};

/*!
 * \brief Create a problem
 * \param lfirst The beginning of the labels
 * \param llast The end of the labels
 * \param ifirst The beginning of the samples
 * \param ilast The end of the samples
 * \param scale true if the problem features must be scaled
 * \return the created problem
 */
template <typename LIterator, typename IIterator>
problem make_problem(LIterator lfirst, LIterator llast, IIterator ifirst, IIterator ilast, bool scale = false) {
    cpp_assert(std::distance(lfirst, llast) == std::distance(ifirst, ilast), "Ranges must be of the same size");
    cpp_unused(ilast); //Ensure no warning is issued for ilast (used only in debug mode)

    auto n_samples = std::distance(lfirst, llast);

    problem problem(n_samples, ifirst->size());

    std::size_t s = 0;

    while (lfirst != llast) {
        problem.label(s) = *lfirst;

        auto features = ifirst->size();

        for (std::size_t i = 0; i < features; ++i) { problem.sample(s)[i].index = i + 1;
            problem.sample(s)[i].value = (*ifirst)[i];
        }

        //End the vector
        problem.sample(s)[features].index = -1;
        problem.sample(s)[features].value = 0.0;

        ++lfirst;
        ++ifirst;
        ++s;
    }

    if (scale) {
        problem.scale();
    }

    return problem;
}

/*!
 * \brief Create a problem
 * \param labels The labels
 * \param samples The samples
 * \param max The maximum number of samples and labels to use
 * \param shuffle true if the data needs to be shuffled
 * \param scale true if the problem features must be scaled
 * \return the created problem
 */
template <typename Labels, typename Images>
problem make_problem(Labels& labels, Images& samples, std::size_t max = 0, bool shuffle = true, bool scale = false) {
    cpp_assert(labels.size() == samples.size(), "There must be the same number of labels and images");

    if (shuffle) {
        static std::random_device rd;
        static std::mt19937_64 g(rd());

        cpp::parallel_shuffle(samples.begin(), samples.end(), labels.begin(), labels.end(), g);
    }

    if (max > 0 && max < labels.size()) {
        labels.resize(max);
        samples.resize(max);
    }

    auto n_samples = labels.size();

    problem problem(n_samples, samples.front().size());

    for (std::size_t s = 0; s < n_samples; ++s) {
        auto features = samples[s].size();

        problem.label(s) = labels[s];

        for (std::size_t i = 0; i < features; ++i) {
            problem.sample(s)[i].index = i + 1;
            problem.sample(s)[i].value = samples[s][i];
        }

        //End the vector
        problem.sample(s)[features].index = -1;
        problem.sample(s)[features].value = 0.0;
    }

    if (scale) {
        problem.scale();
    }

    return problem;
}

/*!
 * \brief Create a problem
 * \param labels The labels
 * \param samples The samples
 * \param scale true if the problem features must be scaled
 * \return the created problem
 */
template <typename Labels, typename Images>
problem make_problem(const Labels& labels, const Images& samples, bool scale = false) {
    return make_problem(labels, samples, 0, false, scale);
}

/*!
 * \brief Generate a default set of parameters
 * \return the set of parameters
 */
inline svm_parameter default_parameters() {
    svm_parameter parameters;

    parameters.svm_type     = C_SVC;
    parameters.kernel_type  = RBF;
    parameters.degree       = 3;
    parameters.gamma        = 0;
    parameters.coef0        = 0;
    parameters.nu           = 0.5;
    parameters.cache_size   = 100;
    parameters.C            = 1;
    parameters.eps          = 1e-3;
    parameters.p            = 0.1;
    parameters.shrinking    = 1;
    parameters.probability  = 0;
    parameters.nr_weight    = 0;
    parameters.weight_label = nullptr;
    parameters.weight       = nullptr;

    return parameters;
}

inline double predict(model& model, const svm_node* sample) {
    std::vector<double> prob_estimates(model.classes());

    return svm_predict_probability(model.get_model(), sample, &prob_estimates[0]);
}

template <typename Sample, cpp::enable_if_u<std::is_convertible<typename Sample::value_type, double>::value> = cpp::detail::dummy>
double predict(model& model, const Sample& sample) {
    std::vector<double> prob_estimates(model.classes());

    auto features = sample.size();
    std::vector<svm_node> svm_sample(features + 1);

    for (std::size_t i = 0; i < features; ++i) {
        svm_sample[i].index = i + 1;
        svm_sample[i].value = sample[i];
    }

    //End the vector
    svm_sample[features].index = -1;
    svm_sample[features].value = 0.0;

    return svm_predict_probability(model.get_model(), &svm_sample[0], &prob_estimates[0]);
}

/*!
 * \brief Test a model on the given problem
 * \param problem The problem description
 * \param model The model to test
 */
inline double test_model(problem& problem, model& model) {
    std::vector<double> prob_estimates(model.classes());

    std::size_t correct = 0;

    for (std::size_t s = 0; s < problem.n_samples; ++s) {
        auto label = svm_predict_probability(model.get_model(), problem.sample(s), &prob_estimates[0]);

        if (label == problem.label(s)) {
            ++correct;
        }
    }

    double accuracy = 100.0 * correct / problem.n_samples;

    std::cout << "Samples: " << problem.n_samples << std::endl;
    std::cout << "Correct: " << correct << std::endl;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    std::cout << "Error: " << (100.0 - accuracy) << "%" << std::endl;

    return accuracy;
}

/*!
 * \brief Train a model for the given problem and parameter set
 * \param problem The problem description
 * \param parameters The training parameters
 * \return The trained model
 */
inline svm_model* train(problem& problem, const svm_parameter& parameters) {
    std::cout << "Train SVM: " << problem.n_samples << " samples" << std::endl;

    auto model = svm_train(&problem.get_problem(), &parameters);

    std::cout << "Training done" << std::endl;

    return model;
}

inline double cross_validate(problem& problem, const svm_parameter& parameters, std::size_t n_fold, bool quiet = false) {
    if (!quiet) {
        std::cout << "Cross validation" << std::endl;
    }

    double* target = new double[problem.n_samples];

    svm_cross_validation(&problem.get_problem(), &parameters, n_fold, target);

    std::size_t cross_correct = 0;

    for (std::size_t i = 0; i < problem.n_samples; ++i) {
        if (target[i] == problem.label(i)) {
            ++cross_correct;
        }
    }

    if (!quiet) {
        std::cout << "Cross validation Samples: " << problem.n_samples << std::endl;
        std::cout << "Cross validation Correct: " << cross_correct << std::endl;
        std::cout << "Cross validation Accuracy: " << (100.0 * cross_correct / problem.n_samples) << "%" << std::endl;
        std::cout << "Cross validation Error: " << (100.0 - (100.0 * cross_correct / problem.n_samples)) << "%" << std::endl;

        std::cout << "Cross validation done" << std::endl;
    }

    delete[] target;

    return 100.0 * cross_correct / problem.n_samples;
}

/*!
 * \brief Check the validity of the problem and parameters
 * \param problem The problem to check
 * \param parameters The parameters to test
 * \return true if it is correct, false otherwise
 */
inline bool check(const problem& problem, const svm_parameter& parameters) {
    auto error = svm_check_parameter(&problem.get_problem(), &parameters);

    if (error) {
        std::cerr << "Parameters not checked: " << error << std::endl;

        return false;
    }

    return true;
}

/*!
 * \brief Load a model from file
 * \param file_name The fiel to load the model from
 * \return The loaded model
 */
inline model load(const std::string& file_name) {
    std::cout << "Load SVM model" << std::endl;

    auto model = svm_load_model(file_name.c_str());

    if (!model) {
        std::cout << "Impossible to load model" << std::endl;
    } else {
        std::cout << "SVM model loaded" << std::endl;
    }

    return {model};
}

/*!
 * \brief Save the given model into the given file
 * \param model The model to save
 * \param file_name The path to the target file
 * \return true if it was saved correctly, false otherwise
 */
inline bool save(const model& model, const std::string& file_name) {
    return !svm_save_model(file_name.c_str(), model.get_model());
}

inline void print_null(const char* /*s*/) {}

/*!
 * \brief Make libsvm quiet during training
 */
inline void make_quiet() {
    svm_set_print_string_function(&print_null);
}

inline void rbf_grid_search(svm::problem& problem, const svm_parameter& parameters, std::size_t n_fold, const std::vector<double>& c_values, const std::vector<double>& gamma_values) {
    std::cout << "Grid Search" << std::endl;
    std::cout << "C in [";
    for (auto C : c_values) {
        std::cout << C << ",";
    }
    std::cout << "]" << std::endl;
    std::cout << "y in [";
    for (auto y : gamma_values) {
        std::cout << y << ",";
    }
    std::cout << "]" << std::endl;

    double max_accuracy   = 0.0;
    std::size_t max_C     = 0;
    std::size_t max_gamma = 0;

    for (auto C : c_values) {
        for (auto gamma : gamma_values) {
            svm_parameter new_parameter = parameters;

            new_parameter.C     = C;
            new_parameter.gamma = gamma;

            auto accuracy = svm::cross_validate(problem, new_parameter, n_fold, true);

            std::cout << "C=" << C << ",y=" << gamma << " -> " << accuracy << std::endl;

            if (accuracy > max_accuracy) {
                max_accuracy = accuracy;
                max_C        = C;
                max_gamma    = gamma;
            }
        }
    }

    std::cout << "Best: C=" << max_C << ",y=" << max_gamma << " -> " << max_accuracy << std::endl;
}

/*!
 * \brief Enumeration for the type of grid search to do
 */
enum class grid_search_type {
    LINEAR, ///< Linear search
    EXP     ///< Exponetial seach
};

/*!
 * \brief Descriptor for the grid of parameters to be searched for an RBF kernel
 */
struct rbf_grid {
    double c_first            = 2e-5;                  ///< The first c to test
    double c_last             = 2e15;                  ///< The last c to test
    std::size_t c_steps       = 10;                    ///< The number of steps for c
    grid_search_type c_search = grid_search_type::EXP; ///< The type of search for c

    double gamma_first            = 2e-15;                 ///< The first gamma to test
    double gamma_last             = 2e3;                   ///< The last gamma to test
    std::size_t gamma_steps       = 10;                    ///< The number of steps for gamma
    grid_search_type gamma_search = grid_search_type::EXP; ///< The type of search for gamma
};

/*!
 * \brief Generate alues for a grid search
 * \param steps The number of steps
 * \param first The first value
 * \param last The last value
 * \param type The type of search
 * \return a vector ontainining all the values of the search
 */
inline std::vector<double> generate_values(std::size_t steps, double first, double last, grid_search_type type) {
    std::vector<double> values(steps);

    if (steps == 1) {
        values[0] = first;
    } else {
        switch (type) {
            case grid_search_type::LINEAR: {
                auto current = first;
                for (std::size_t i = 0; i < steps; ++i) {
                    values[i] = current;
                    current += (last - first) / (steps - 1);
                }
            }

            break;

            case grid_search_type::EXP: {
                auto current = first;
                for (std::size_t i = 0; i < steps; ++i) {
                    values[i] = current;
                    current *= std::pow(last / first, 1.0 / (steps - 1));
                }
            }

            break;
        }
    }

    return values;
}

inline void rbf_grid_search(svm::problem& problem, const svm_parameter& parameters, std::size_t n_fold, const rbf_grid& g = rbf_grid()) {
    auto c_values     = generate_values(g.c_steps, g.c_first, g.c_last, g.c_search);
    auto gamma_values = generate_values(g.gamma_steps, g.gamma_first, g.gamma_last, g.gamma_search);

    rbf_grid_search(problem, parameters, n_fold, c_values, gamma_values);
}

} //end of namespace svm

#endif
