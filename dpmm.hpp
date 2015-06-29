/*
 * Dirichlet process mixture model with local MAP or particle filter
 * approximation. Also called the rational model of categorisation. The model
 * specifications follow the description by Sanborn, Griffiths, and Navarro
 * (2010).
 *
 * Only discrete dimensions are supported.
 *
 * Sanborn, A. N., Griffiths, T. L., & Navarro, D. J. (2010).  Rational
 * approximations to rational models: alternative algorithms for category
 * learning. Psychological Review, 117, 1144–1167.
 */


#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <assert.h>


namespace DPMM
{

// for comparing double variables. If their difference is less than this value, they are
// treated equal.
extern const double PRACTICAL_ZERO;


namespace Misc
{

/*
 * Not really part of the model, but handy functions to use with the model.
*/

double generate_random_double(void);

void get_unique_values(const std::vector<std::vector<unsigned int>>& objects,
                       std::vector<std::set<unsigned int>>& unique_values,
                       std::vector<unsigned int>& n_uniques);

}  // namespace Misc


enum class Approximation_Method
{

    local_map,
    particle_filter

};  // enum class Approximation_Method


class Parameter
{

  public:
    Parameter(void){};

    Parameter(const double coupling_parameter, const std::vector<double> beta,
              const Approximation_Method approximation_method,
              const unsigned int n_particles = 1)
        : c(coupling_parameter), beta(beta), approximation_method(approximation_method),
          n_particles(n_particles){};

    Parameter& operator=(const Parameter& p);

    std::string get_approximation_method_name(void);

    double c;
    std::vector<double> beta;
    Approximation_Method approximation_method;
    unsigned int n_particles;

};  // class Parameter


class Cluster
{

  public:
    Cluster(const Parameter& param, const std::vector<unsigned int>& n_uniques);
    Cluster(const Cluster& c);  // to copy class
    Cluster(Cluster&& c);       // to move class
    ~Cluster(void);

    Cluster& operator=(const Cluster& c);
    Cluster& operator=(Cluster&& c);

    // p(category | cluster)
    double predict(const std::vector<unsigned int>& object,
                   const unsigned int dimension_to_be_predicted);

    // unnormalised p(cluster | object)
    double compute_unnormalised_posterior(
        const unsigned int n_observed, const std::vector<unsigned int>& object,
        const std::vector<unsigned int>& visible_dimensions);

    // add an object to cluster
    void add_object(const std::vector<unsigned int>& object,
                    const std::vector<unsigned int>& visible_dimensions);

    // p(cluster)
    double compute_prior(const unsigned int n_observed);

    // H(object)
    // Dimensions are assumed independent, so entropy per dimensions are summed.
    double compute_entropy_object(
        const std::vector<unsigned int>& visible_dimensions,
        const std::vector<std::set<unsigned int>>& unique_values);

    // variables needed to be public for copying
    unsigned int n_members;
    // vector of objects in cluster
    std::vector<std::vector<unsigned int>> members;

  private:
    const Parameter& parameter;

    // number of possible values for each dimension
    const std::vector<unsigned int>& n_uniques;

    // practical nan value for dimension value
    const unsigned int nanvalue = std::numeric_limits<unsigned int>::max();

    void copy_cluster(const Cluster& c);

    // p(object | cluster)
    double compute_likelihood(const std::vector<unsigned int>& object,
                              const std::vector<unsigned int>& visible_dimensions);
    double compute_likelihood_one_discrete_feature(
        const std::vector<unsigned int>& object, const unsigned int dimension);

};  // class Cluster


class Particle
{

  public:
    Particle(const Parameter& parameter, const std::vector<unsigned int>& n_uniques);
    Particle(const Particle& p);  // copy
    Particle(Particle&& p);       // move
    ~Particle();

    Particle& operator=(const Particle& p);  // copy
    Particle& operator=(Particle&& p);       // move

    // predict without assiging an object to a cluster
    // predictions from each cluster are weighted and averaged
    double predict(const std::vector<unsigned int>& object,
                   const std::vector<unsigned int>& visible_dimensions,
                   const unsigned int dimension_to_be_predicted);

    // assign an object to a cluster
    void learn(const std::vector<unsigned int>& object,
               const std::vector<unsigned int>& visible_dimensions);

    // sum of cluster weights
    double get_particle_weight(void);

    // H(cluster | particle)
    double compute_cluster_entropy(void);

    // compute mutual information, I(object; cluster | particle)
    double compute_cluster_informativeness(
        const std::vector<unsigned int>& visible_dimensions,
        const std::vector<std::set<unsigned int>>& unique_values);

    // variables needed to be public for copying
    const Parameter& parameter;
    unsigned int n_observed;
    std::vector<Cluster> clusters;
    std::vector<double> cluster_weights;
    unsigned int n_clusters;

  private:
    const std::vector<unsigned int>& n_uniques;

    // index of cluster to which an object is assigned
    unsigned int winning_cluster;

    void copy_particle(const Particle& p);

    // compute unnormalised posterior for each cluster
    void compute_cluster_weights(const std::vector<unsigned int>& object,
                                 const std::vector<unsigned int>& visible_dimensions);

    // decide which cluster an object is assigned to
    void find_winner(void);
    void find_winner_softmax(void);  // for particle filter
    void find_winner_map(void);      // for local map

};  // class Particle


class Model
{

  public:
    Model(void);
    Model(const Parameter& parameter, const std::vector<unsigned int>& n_uniques);
    ~Model();

    // assign an object to a cluster and re-sample particles
    void learn(const std::vector<unsigned int>& object,
               const std::vector<unsigned int>& visible_dimensions);
    void learn(const std::vector<std::vector<unsigned int>>& object,
               const std::vector<unsigned int>& visible_dimensions);
    void learn(const std::vector<std::vector<unsigned int>>& object,
               const std::vector<std::vector<unsigned int>>& visible_dimensions);

    // predict without assigning an object to a cluster
    void predict(const std::vector<unsigned int>& object,
                 const std::vector<unsigned int>& visible_dimensions,
                 const unsigned int dimension_to_be_predicted, double& accuracy);
    void predict(const std::vector<std::vector<unsigned int>>& objects,
                 const std::vector<unsigned int>& visible_dimensions,
                 const unsigned int dimension_to_be_predicted,
                 std::vector<double>& accuracy);
    void predict(const std::vector<std::vector<unsigned int>>& objects,
                 const std::vector<unsigned int>& visible_dimensions,
                 const std::vector<unsigned int>& dimension_to_be_predicted,
                 std::vector<double>& accuracy);

    // summary statistics for cluster: n, informativeness, and entropy.
    //  n: weighted average number of clusters
    //  informativeness: weighted average of mutual information I(object; entropy)
    //  entropy: weighted average of entropy H(cluster)
    void get_cluster_statistics(const std::vector<unsigned int>& visible_dimensions,
                                const std::vector<std::set<unsigned int>>& unique_values,
                                std::map<std::string, double>& stats);

    // clear out whatever has been learnt
    void forget(void);
    // re-set parameter and n_uniques. also forget
    void reset(const Parameter& new_parameter,
               const std::vector<unsigned int>& new_n_uniques);

    // accessed from test_dmpp
    std::vector<Particle> particles;

    Parameter parameter;

  private:
    std::vector<unsigned int> n_uniques;

    // create particles
    void initialise_particles(void);

    // p(particle | observed objects)
    std::vector<double> particle_weights;
    void compute_particle_weights(void);

    void resample_particles(void);

    // mean number of clusters
    double get_n_clusters(void);
    // I(object; cluster)
    double get_cluster_informativeness(
        const std::vector<unsigned int>& visible_dimensions,
        const std::vector<std::set<unsigned int>>& unique_values);
    // H(cluster)
    double get_cluster_entropy(void);

};  // class Model


}  // namespace DPMM
