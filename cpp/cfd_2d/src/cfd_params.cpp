#include "cfd.h"

using namespace CFD;

SolverType CFD::convertSolverType(const std::string& solver) {
    if (solver == "jacobi") {
        return SolverType::JACOBI;
    }
    else if (solver == "multigrid_jacobi") {
        return SolverType::MULTIGRID_JACOBI;
    }
    else if (solver == "conjugate_gradient") {
        return SolverType::CONJUGATE_GRADIENT;
    }
    else if (solver == "jpcg") {
        return SolverType::JPCG;
    }
    else if (solver == "apcg") {
        return SolverType::APCG;
    }
    else if (solver == "mgpcg") {
        return SolverType::MGPCG;
    }
    else if (solver == "mgpsd") {
        return SolverType::MGPSD;
    }
    else if (solver == "sd") {
        return SolverType::STEEPEST_DESCENT;
    }
    else if (solver == "ml") {
        return SolverType::ML;
    }
    else if (solver == "richardson") {
        return SolverType::RICHARDSON;
    }
    else if (solver == "dcdm") {
        return SolverType::DCDM;
    }
    else if (solver == "omgpcg") {
        return SolverType::OMGPCG;
    }
    else {
        throw std::invalid_argument("Invalid solver type");
    }
}

FluidParams::FluidParams(std::string name, int argc, char* argv[]) 
    : argument_parser(name)
{
    this->argument_parser.add_argument("-i", "--imax").help("imax").default_value(this->imax).action([](const std::string& value) { return std::stoi(value); });
    this->argument_parser.add_argument("-j", "--jmax").help("jmax").default_value(this->jmax).action([](const std::string& value) { return std::stoi(value); });
    this->argument_parser.add_argument("-x", "--xlength").help("xlength").default_value(this->xlength).action([](const std::string& value) { return std::stof(value); });
    this->argument_parser.add_argument("-y", "--ylength").help("ylength").default_value(this->ylength).action([](const std::string& value) { return std::stof(value); });
    this->argument_parser.add_argument("-z", "--t_end").help("t_end").default_value(this->t_end).action([](const std::string& value) { return std::stof(value); });
    this->argument_parser.add_argument("-u", "--tau").help("tau").default_value(this->tau).action([](const std::string& value) { return std::stof(value); });
    this->argument_parser.add_argument("-e", "--eps").help("eps").default_value(this->eps).action([](const std::string& value) { return std::stof(value); });
    this->argument_parser.add_argument("-o", "--omg").help("omg").default_value(this->omg).action([](const std::string& value) { return std::stof(value); });
    this->argument_parser.add_argument("-a", "--alpha").help("alpha").default_value(this->alpha).action([](const std::string& value) { return std::stof(value); });
    this->argument_parser.add_argument("-r", "--Re").help("Re").default_value(this->Re).action([](const std::string& value) { return std::stof(value); });
    this->argument_parser.add_argument("-t", "--t").help("t").default_value(this->t).action([](const std::string& value) { return std::stof(value); });
    this->argument_parser.add_argument("-d", "--dt").help("dt").default_value(this->dt).action([](const std::string& value) { return std::stof(value); });
    this->argument_parser.add_argument("-l", "--save_interval").help("VTK save interval").default_value(this->save_interval).action([](const std::string& value) { return std::stof(value); });
    this->argument_parser.add_argument("-s", "--solver").help("solver").default_value("jacobi").action([](const std::string& value) { return value; });
    this->argument_parser.add_argument("--save_ml").help("Save for Machine Learning").default_value(false).action([](const std::string& value) { return std::stoi(value) == 1; });
    this->argument_parser.add_argument("--no_vtk").help("Disable VTK Rendering").default_value(false).action([](const std::string& value) { return std::stoi(value) == 1; });
    this->argument_parser.add_argument("--num_sweeps").help("Number of sweeps for multigrid").default_value(this->num_sweeps).action([](const std::string& value) { return std::stof(value); });
    this->argument_parser.add_argument("--ml_model_path").help("Path to ML model").default_value(this->ml_model_path).action([](const std::string& value) { return value; });
    this->argument_parser.add_argument("--safety_factor").help("Safety factor for ML").default_value(this->safety_factor).action([](const std::string& value) { return std::stof(value); });
    this->argument_parser.add_argument("--radius").help("Radius for obstacle").default_value(this->radius).action([](const std::string& value) { return std::stof(value); });


    try {
        this->argument_parser.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << this->argument_parser;
        exit(1);
    }

    this->imax = this->argument_parser.get<int>("--imax"),
    this->jmax = this->argument_parser.get<int>("--jmax"),
    this->xlength = this->argument_parser.get<float>("--xlength"),
    this->ylength = this->argument_parser.get<float>("--ylength"),
    this->t_end = this->argument_parser.get<float>("--t_end"),
    this->tau = this->argument_parser.get<float>("--tau"),
    this->eps = this->argument_parser.get<float>("--eps"),
    this->omg = this->argument_parser.get<float>("--omg"),
    this->alpha = this->argument_parser.get<float>("--alpha"),
    this->Re = this->argument_parser.get<float>("--Re"),
    this->t = this->argument_parser.get<float>("--t"),
    this->dt = this->argument_parser.get<float>("--dt"),
    this->save_interval = this->argument_parser.get<float>("--save_interval"),
    this->save_ml = this->argument_parser.get<bool>("--save_ml"),
    this->no_vtk = this->argument_parser.get<bool>("--no_vtk"),
    this->solver_type = convertSolverType(this->argument_parser.get<std::string>("--solver"));
    this->num_sweeps = this->argument_parser.get<float>("--num_sweeps");
    this->ml_model_path = this->argument_parser.get<std::string>("--ml_model_path");
    this->safety_factor = this->argument_parser.get<float>("--safety_factor");
    this->radius = this->argument_parser.get<float>("--radius");

    this->argc = argc;
    this->argv = argv;
}