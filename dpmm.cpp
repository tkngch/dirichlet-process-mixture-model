#include "dpmm.hpp"


namespace DPMM
{

const double PRACTICAL_ZERO = 1e-10;


namespace Misc
{

const unsigned SEED = 1234567890;
std::mt19937 RANDOM_NUMBER_GENERATOR{SEED};
std::uniform_real_distribution<double> sample_uniform(0.0, 1.0);
auto generate_random_double_ = std::bind(sample_uniform, RANDOM_NUMBER_GENERATOR);

double generate_random_double(void) { return generate_random_double_(); }


void get_unique_values(const std::vector<std::vector<unsigned int>>& objects,
                       std::vector<std::set<unsigned int>>& unique_values,
                       std::vector<unsigned int>& n_uniques)
{

    const unsigned int n_dims = objects.at(0).size();

    unique_values.resize(n_dims);
    n_uniques.resize(n_dims);

    for (unsigned int dim = 0; dim < n_dims; dim++)
    {
        unique_values.at(dim).clear();

        for (auto object : objects)
        {
            unique_values.at(dim).insert(object.at(dim));
        }

        n_uniques.at(dim) = unique_values.at(dim).size();
    }
}

}  // namespace Misc


Parameter& Parameter::operator=(const Parameter& p)
{
    c = p.c;
    beta = p.beta;
    approximation_method = p.approximation_method;
    n_particles = p.n_particles;

    return *this;
}


std::string Parameter::get_approximation_method_name(void)
{
    if (approximation_method == Approximation_Method::local_map)
    {
        return std::string("Local MAP");
    }
    else if (approximation_method == Approximation_Method::particle_filter)
    {
        return std::string("Particle Filter");
    }
    else
    {
        return std::string("Unknown Method");
    }
}


Cluster::Cluster(const Parameter& parameter, const std::vector<unsigned int>& n_uniques)
    : parameter(parameter), n_uniques(n_uniques)
{
    n_members = 0;
    assert(n_uniques.size() == parameter.beta.size());
}


Cluster::Cluster(const Cluster& c) : parameter(c.parameter), n_uniques(c.n_uniques)
{
    copy_cluster(c);
}


Cluster::Cluster(Cluster&& c) : parameter(c.parameter), n_uniques(std::move(c.n_uniques))
{
    members = std::move(c.members);
    n_members = c.n_members;
    assert(n_uniques.size() == parameter.beta.size());
}


Cluster::~Cluster() {}


Cluster& Cluster::operator=(const Cluster& c)
{
    copy_cluster(c);
    return *this;
}


Cluster& Cluster::operator=(Cluster&& c)
{
    if (this != &c)
    {
        assert(n_uniques == c.n_uniques);

        members = std::move(c.members);
        n_members = c.n_members;
    }
    return *this;
}


void Cluster::copy_cluster(const Cluster& c)
{
    assert(n_uniques == c.n_uniques);

    members.clear();
    members = c.members;
    n_members = c.n_members;
}


double Cluster::predict(const std::vector<unsigned int>& object,
                        const unsigned int dimension_to_be_predicted)
{
    double prediction;
    assert(n_uniques.at(dimension_to_be_predicted) > 0);

    prediction =
        compute_likelihood_one_discrete_feature(object, dimension_to_be_predicted);

    assert(0 <= prediction);
    assert(prediction <= 1);

    return prediction;
}


double Cluster::compute_unnormalised_posterior(
    const unsigned int n_observed, const std::vector<unsigned int>& object,
    const std::vector<unsigned int>& visible_dimensions)
{
    const double prior = compute_prior(n_observed);
    const double likelihood = compute_likelihood(object, visible_dimensions);
    const double unnormalised_posterior = prior * likelihood;

    return unnormalised_posterior;
}


void Cluster::add_object(const std::vector<unsigned int>& object,
                         const std::vector<unsigned int>& visible_dimensions)
{
    std::vector<unsigned int> object_to_be_added(object.size(), nanvalue);

    for (auto dimension : visible_dimensions)
    {
        assert(object.at(dimension) != nanvalue);
        object_to_be_added.at(dimension) = object.at(dimension);
    }

    members.push_back(object_to_be_added);
    n_members++;
}


double Cluster::compute_entropy_object(
    const std::vector<unsigned int>& visible_dimensions,
    const std::vector<std::set<unsigned int>>& unique_values)
{
    double entropy = 0;
    double p;
    std::vector<unsigned int> test_object;
    test_object.resize(
        1 + *std::max_element(begin(visible_dimensions), end(visible_dimensions)));

    for (unsigned int dim : visible_dimensions)
    {
        for (unsigned int value : unique_values.at(dim))
        {
            test_object.at(dim) = value;
            p = compute_likelihood_one_discrete_feature(test_object, dim);

            if (p > 0)
            {
                entropy += -1 * p * log2(p);
            }
        }
    }

    return entropy;
}


double Cluster::compute_prior(const unsigned int n_observed)
{
    double prior;

    if (parameter.c < PRACTICAL_ZERO)
    {
        prior = (double)(n_members == 0);
    }
    else
    {
        const double alpha = (1 - parameter.c) / parameter.c;

        if (n_members == 0)
        {
            prior = alpha / (n_observed - 1.0 + alpha);
        }
        else
        {
            prior = n_members / (n_observed - 1.0 + alpha);
        }
    }

    return prior;
}


double Cluster::compute_likelihood(const std::vector<unsigned int>& object,
                                   const std::vector<unsigned int>& visible_dimensions)
{
    double likelihood = 1;
    for (auto dimension : visible_dimensions)
    {

        assert(n_uniques.at(dimension) > 0);
        assert(object.at(dimension) != nanvalue);

        likelihood *= compute_likelihood_one_discrete_feature(object, dimension);
    }

    return likelihood;
}


double Cluster::compute_likelihood_one_discrete_feature(
    const std::vector<unsigned int>& object, const unsigned int dimension)
{
    unsigned int n_match = 0, n_total = 0;
    for (auto member : members)
    {
        if (member.at(dimension) != nanvalue)
        {
            if (object.at(dimension) == member.at(dimension))
            {
                n_match++;
            }

            n_total++;
        }
    }

    const double likelihood =
        (((double)n_match + parameter.beta.at(dimension)) /
         ((double)n_total + parameter.beta.at(dimension) * n_uniques.at(dimension)));

    return likelihood;
}


Particle::Particle(const Parameter& parameter, const std::vector<unsigned int>& n_uniques)
    : parameter(parameter), n_uniques(n_uniques)
{
    n_observed = 0;

    clusters.clear();
    clusters.push_back({parameter, n_uniques});

    cluster_weights.clear();
    cluster_weights.resize(1);

    n_clusters = 1;
}


Particle::Particle(const Particle& p) : parameter(p.parameter), n_uniques(p.n_uniques)
{
    copy_particle(p);
}


Particle::Particle(Particle&& p) : parameter(p.parameter), n_uniques(p.n_uniques)
{
    n_observed = p.n_observed;

    n_clusters = p.clusters.size();
    clusters = std::move(p.clusters);
    cluster_weights = std::move(p.cluster_weights);
}


Particle::~Particle() {}


Particle& Particle::operator=(const Particle& p)
{
    copy_particle(p);
    return *this;
}


Particle& Particle::operator=(Particle&& p)
{
    if (this != &p)
    {
        // assert(param == p.param);
        assert(n_uniques == p.n_uniques);

        n_observed = p.n_observed;

        n_clusters = p.clusters.size();
        clusters = std::move(p.clusters);
        cluster_weights = std::move(p.cluster_weights);
    }
    return *this;
}


void Particle::copy_particle(const Particle& p)
{
    assert(n_uniques == p.n_uniques);

    n_observed = p.n_observed;
    n_clusters = p.n_clusters;

    clusters.clear();
    clusters = p.clusters;

    cluster_weights.clear();
    cluster_weights = p.cluster_weights;
}


double Particle::predict(const std::vector<unsigned int>& object,
                         const std::vector<unsigned int>& visible_dimensions,
                         const unsigned int dimension_to_be_predicted)
{
    n_observed++;
    compute_cluster_weights(object, visible_dimensions);

    double prediction = 0;
    const double normaliser = get_particle_weight();
    for (unsigned int i = 0; i < n_clusters; i++)
    {
        if (cluster_weights.at(i) > PRACTICAL_ZERO)
        {
            prediction += (cluster_weights.at(i) / normaliser) *
                          clusters.at(i).predict(object, dimension_to_be_predicted);
        }
    }

    // an object is not assigned to a cluster. revert the count
    n_observed--;

    assert(0 <= prediction);
    assert(prediction <= 1);
    return prediction;
}


void Particle::learn(const std::vector<unsigned int>& object,
                     const std::vector<unsigned int>& visible_dimensions)
{
    n_observed++;

    compute_cluster_weights(object, visible_dimensions);
    find_winner();

    if (winning_cluster == n_clusters - 1)
    {
        clusters.push_back({parameter, n_uniques});
        n_clusters = clusters.size();
    }
    clusters.at(winning_cluster).add_object(object, visible_dimensions);
}


double Particle::get_particle_weight(void)
{
    const double particle_weight =
        std::accumulate(begin(cluster_weights), end(cluster_weights), 0.0);

    return particle_weight;
}


double Particle::compute_cluster_entropy(void)
{
    n_observed++;
    double p, entropy = 0;
    for (auto cluster : clusters)
    {
        p = cluster.compute_prior(n_observed);
        if (p > 0)
        {
            entropy += -1 * p * log2(p);
        }
    }
    n_observed--;
    return entropy;
}


double Particle::compute_cluster_informativeness(
    const std::vector<unsigned int>& visible_dimensions,
    const std::vector<std::set<unsigned int>>& unique_values)
{
    n_observed++;
    // I(object; cluster) = H(object) - H(object | cluster)

    // H(object): entropy with prior, which is the entropy with zero-member cluster.
    assert(clusters.at(n_clusters - 1).n_members == 0);
    double mutual_info = clusters.at(n_clusters - 1)
                             .compute_entropy_object(visible_dimensions, unique_values);

    // H(object | cluster)
    for (auto cluster : clusters)
    {
        mutual_info -= cluster.compute_prior(n_observed) *
                       cluster.compute_entropy_object(visible_dimensions, unique_values);
    }
    n_observed--;
    return mutual_info;
}


void Particle::compute_cluster_weights(
    const std::vector<unsigned int>& object,
    const std::vector<unsigned int>& visible_dimensions)
{
    assert(n_clusters > 0);

    cluster_weights.resize(n_clusters);
    for (unsigned int i = 0; i < n_clusters; i++)
    {
        cluster_weights.at(i) = clusters.at(i).compute_unnormalised_posterior(
            n_observed, object, visible_dimensions);
    }
}


void Particle::find_winner(void)
{
    if (n_clusters == 1)
    {
        winning_cluster = 0;
    }
    else if (parameter.approximation_method == Approximation_Method::particle_filter)
    {
        find_winner_softmax();
    }
    else if (parameter.approximation_method == Approximation_Method::local_map)
    {
        find_winner_map();
    }

    assert(winning_cluster < n_clusters);
}


void Particle::find_winner_softmax(void)
{
    const double normaliser =
        std::accumulate(begin(cluster_weights), end(cluster_weights), 0.0);
    double u = Misc::generate_random_double();
    double p;

    winning_cluster = n_clusters;

    for (unsigned int i = 0; i < n_clusters; i++)
    {
        p = cluster_weights[i] / normaliser;

        if (u < p)
        {
            winning_cluster = i;
            u = -100;
            break;
        }
        else
        {
            u -= p;
        }
    }
}


void Particle::find_winner_map(void)
{
    double current_max = 0;
    std::vector<unsigned int> winners;

    winning_cluster = n_clusters;

    for (unsigned int i = 0; i < n_clusters; i++)
    {

        if (fabs(cluster_weights.at(i) - current_max) < PRACTICAL_ZERO)
        {
            winners.push_back(i);
        }
        else if (cluster_weights.at(i) > current_max)
        {
            winners.clear();
            winners.push_back(i);
            current_max = cluster_weights.at(i);
        }
    }

    std::random_shuffle(begin(winners), end(winners));
    winning_cluster = winners.at(0);
}


Model::Model(void) {}


Model::Model(const Parameter& parameter, const std::vector<unsigned int>& n_uniques)
    : parameter(parameter), n_uniques(n_uniques)
{
    // one beta for one dimension
    assert(n_uniques.size() == parameter.beta.size());

    initialise_particles();
}


Model::~Model() {}


void Model::forget(void)
{
    particles.clear();
    initialise_particles();
}

void Model::reset(const Parameter& new_parameter,
                  const std::vector<unsigned int>& new_n_uniques)
{
    // one beta for one dimension
    assert(new_n_uniques.size() == new_parameter.beta.size());

    parameter = new_parameter;
    n_uniques = new_n_uniques;

    forget();
}


void Model::learn(const std::vector<unsigned int>& object,
                  const std::vector<unsigned int>& visible_dimensions)
{
    for (unsigned int i = 0; i < parameter.n_particles; i++)
    {
        particles.at(i).learn(object, visible_dimensions);
    }

    if (parameter.n_particles > 1)
    {
        resample_particles();
    }
}


void Model::learn(const std::vector<std::vector<unsigned int>>& objects,
                  const std::vector<unsigned int>& visible_dimensions)
{
    for (auto object : objects)
    {
        learn(object, visible_dimensions);
    }
}


void Model::learn(const std::vector<std::vector<unsigned int>>& objects,
                  const std::vector<std::vector<unsigned int>>& visible_dimensions)
{
    assert(objects.size() == visible_dimensions.size());

    const unsigned int n_objects = objects.size();
    for (unsigned int i = 0; i < n_objects; i++)
    {
        learn(objects.at(i), visible_dimensions.at(i));
    }
}


void Model::predict(const std::vector<unsigned int>& object,
                    const std::vector<unsigned int>& visible_dimensions,
                    const unsigned int dimension_to_be_predicted, double& accuracy)
{
    compute_particle_weights();

    accuracy = 0;

    double acc;
    for (unsigned int i = 0; i < parameter.n_particles; i++)
    {
        acc = particles.at(i)
                  .predict(object, visible_dimensions, dimension_to_be_predicted);

        accuracy += acc / parameter.n_particles;  // mean-average
    }

    assert(0 <= accuracy);
    assert(accuracy <= 1);
}


void Model::predict(const std::vector<std::vector<unsigned int>>& objects,
                    const std::vector<unsigned int>& visible_dimensions,
                    const unsigned int dimension_to_be_predicted,
                    std::vector<double>& accuracy)
{

    const unsigned int n_objects = objects.size();
    accuracy.resize(n_objects);

    for (unsigned int t = 0; t < n_objects; t++)
    {
        predict(objects.at(t), visible_dimensions, dimension_to_be_predicted,
                accuracy.at(t));
    }
}


void Model::predict(const std::vector<std::vector<unsigned int>>& objects,
                    const std::vector<unsigned int>& visible_dimensions,
                    const std::vector<unsigned int>& dimension_to_be_predicted,
                    std::vector<double>& accuracy)
{

    assert(objects.size() == dimension_to_be_predicted.size());

    const unsigned int n_objects = objects.size();
    accuracy.resize(n_objects);

    for (unsigned int t = 0; t < n_objects; t++)
    {
        predict(objects.at(t), visible_dimensions, dimension_to_be_predicted.at(t),
                accuracy.at(t));
    }
}


void Model::initialise_particles(void)
{
    for (unsigned int i = 0; i < parameter.n_particles; i++)
    {
        particles.push_back({parameter, n_uniques});
    }
}


void Model::get_cluster_statistics(
    const std::vector<unsigned int>& visible_dimensions,
    const std::vector<std::set<unsigned int>>& unique_values,
    std::map<std::string, double>& stats)
{
    stats["n"] = get_n_clusters();
    stats["informativeness"] =
        get_cluster_informativeness(visible_dimensions, unique_values);
    stats["entropy"] = get_cluster_entropy();
}


double Model::get_n_clusters(void)
{
    double out = 0;
    for (unsigned int i = 0; i < parameter.n_particles; i++)
    {
        // mean-average
        out += (particles.at(i).n_clusters - 1.0) /  // -1 to remove an empty cluster
               parameter.n_particles;
    }

    return out;
}


double Model::get_cluster_informativeness(
    const std::vector<unsigned int>& visible_dimensions,
    const std::vector<std::set<unsigned int>>& unique_values)
{
    double out = 0;
    for (unsigned int i = 0; i < parameter.n_particles; i++)
    {
        // mean-average
        out += particles.at(i)
                   .compute_cluster_informativeness(visible_dimensions, unique_values) /
               parameter.n_particles;
    }
    return out;
}


double Model::get_cluster_entropy(void)
{
    double out = 0;
    for (unsigned int i = 0; i < parameter.n_particles; i++)
    {
        // mean-average
        out += particles.at(i).compute_cluster_entropy() / parameter.n_particles;
    }

    return out;
}


void Model::compute_particle_weights(void)
{
    particle_weights.resize(parameter.n_particles);
    for (unsigned int i = 0; i < parameter.n_particles; i++)
    {
        particle_weights.at(i) = particles.at(i).get_particle_weight();
    }

    const double normaliser =
        std::accumulate(begin(particle_weights), end(particle_weights), 0.0);
    std::transform(begin(particle_weights), end(particle_weights),
                   begin(particle_weights), [&](double x)
                   {
                       return x / normaliser;
                   });
}


void Model::resample_particles(void)
{
    // multinomial samples from particle weight distribution
    compute_particle_weights();

    std::vector<Particle> new_particles;
    new_particles.reserve(parameter.n_particles);

    double u;
    unsigned int j;
    for (unsigned int i = 0; i < parameter.n_particles; i++)
    {
        j = 0;
        u = Misc::generate_random_double();
        do
        {
            if (u < particle_weights[j])
            {
                new_particles.push_back(particles.at(j));
                u = 100;
                break;
            }
            else
            {
                u -= particle_weights.at(j);
                j++;
            }
        } while ((0 <= u) && (u <= 1));
    }
    particles.clear();
    particles = std::move(new_particles);

    assert(particles.size() == parameter.n_particles);
}


}  // namespace DPMM
