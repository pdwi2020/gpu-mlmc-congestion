/* ns3_bottleneck.cc
 *
 * ns-3 packet-level simulation of a single bottleneck link.
 * Validates TV-sigma SDE predictions (P95/P99 queue occupancy) against
 * a packet-level discrete-event reference.
 *
 * Model:
 *   - Poisson packet arrivals at rate λ pkt/s
 *   - Constant service time 1/μ s  (deterministic service, i.e., D/D/1 approx)
 *   - Utilisation ρ = λ/μ ∈ {0.5, 0.7, 0.8, 0.9}
 *   - 10 seeds per ρ; collect P95/P99 queue occupancy
 *
 * Build (ns-3.40+):
 *   Copy this file to <ns3-root>/scratch/ns3_bottleneck.cc
 *   cd <ns3-root> && ./ns3 build
 *   ./ns3 run scratch/ns3_bottleneck
 *
 * Output:
 *   results/ns3/bottleneck_results.csv
 *   columns: rho, seed, p95_queue, p99_queue, mean_queue
 *
 * Compile-time dependencies: ns3 core, network, internet, point-to-point,
 *   applications modules.
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Ns3Bottleneck");

/* -------------------------------------------------------------------------
 * Queue-length recorder: samples queue size every SAMPLE_INTERVAL seconds
 * ------------------------------------------------------------------------- */
static const double SAMPLE_INTERVAL = 0.01;  // 10 ms sampling

struct QueueSampler {
    Ptr<Queue<Packet>> queue;
    std::vector<uint32_t> samples;
    EventId event;

    void Sample () {
        samples.push_back (queue->GetNPackets ());
        event = Simulator::Schedule (Seconds (SAMPLE_INTERVAL),
                                     &QueueSampler::Sample, this);
    }

    double Percentile (double pct) const {
        if (samples.empty ()) return 0.0;
        std::vector<uint32_t> sorted = samples;
        std::sort (sorted.begin (), sorted.end ());
        size_t idx = static_cast<size_t> (std::ceil (pct / 100.0 * sorted.size ())) - 1;
        idx = std::min (idx, sorted.size () - 1);
        return static_cast<double> (sorted[idx]);
    }

    double Mean () const {
        if (samples.empty ()) return 0.0;
        double sum = 0.0;
        for (auto v : samples) sum += v;
        return sum / samples.size ();
    }
};

/* -------------------------------------------------------------------------
 * Run one simulation instance
 * ------------------------------------------------------------------------- */
struct SimResult {
    double rho;
    uint32_t seed;
    double p95;
    double p99;
    double mean;
};

SimResult RunSim (double rho, double mu_rate, uint32_t seed,
                  double sim_time = 60.0, uint32_t pkt_size = 1000)
{
    RngSeedManager::SetSeed (seed);
    RngSeedManager::SetRun  (seed);

    double lambda_rate = rho * mu_rate;            // pkt/s
    double link_bw_bps = mu_rate * pkt_size * 8.0; // bits/s to match service rate
    std::string bw_str = std::to_string (static_cast<uint64_t> (link_bw_bps)) + "bps";

    /* Topology: sender -- bottleneck-link -- sink */
    NodeContainer nodes;
    nodes.Create (2);

    PointToPointHelper p2p;
    p2p.SetDeviceAttribute  ("DataRate", StringValue (bw_str));
    p2p.SetChannelAttribute ("Delay",    StringValue ("1ms"));
    p2p.SetQueue ("ns3::DropTailQueue",
                  "MaxSize", StringValue ("10000p")); // effectively infinite buffer

    NetDeviceContainer devs = p2p.Install (nodes);

    InternetStackHelper stack;
    stack.Install (nodes);

    Ipv4AddressHelper addr;
    addr.SetBase ("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer iface = addr.Assign (devs);

    /* UDP Poisson source on node 0 → node 1 */
    uint16_t port = 9;
    PacketSinkHelper sink ("ns3::UdpSocketFactory",
                           InetSocketAddress (Ipv4Address::GetAny (), port));
    ApplicationContainer sinkApp = sink.Install (nodes.Get (1));
    sinkApp.Start (Seconds (0.0));
    sinkApp.Stop  (Seconds (sim_time + 1.0));

    /* Custom Poisson ON/OFF source mimicking exponential inter-arrivals */
    OnOffHelper onoff ("ns3::UdpSocketFactory",
                       InetSocketAddress (iface.GetAddress (1), port));
    /* OnOffApplication with exponential on/off ~ Poisson(lambda_rate) at packet level */
    double mean_on  = 1.0 / lambda_rate;  // mean ON duration (one packet burst)
    double mean_off = 0.0;                // zero off time → continuous Poisson
    onoff.SetAttribute ("OnTime",
        StringValue ("ns3::ExponentialRandomVariable[Mean=" +
                     std::to_string (mean_on) + "]"));
    onoff.SetAttribute ("OffTime",
        StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
    onoff.SetAttribute ("DataRate", StringValue (bw_str));
    onoff.SetAttribute ("PacketSize", UintegerValue (pkt_size));
    ApplicationContainer srcApp = onoff.Install (nodes.Get (0));
    srcApp.Start (Seconds (0.1));
    srcApp.Stop  (Seconds (sim_time));

    /* Attach queue sampler to the bottleneck device */
    Ptr<PointToPointNetDevice> ppDev =
        DynamicCast<PointToPointNetDevice> (devs.Get (0));
    Ptr<Queue<Packet>> q = ppDev->GetQueue ();

    QueueSampler sampler;
    sampler.queue = q;
    sampler.event = Simulator::Schedule (Seconds (1.0), &QueueSampler::Sample, &sampler);

    Simulator::Stop (Seconds (sim_time + 0.5));
    Simulator::Run ();
    Simulator::Destroy ();

    return {rho, seed, sampler.Percentile (95), sampler.Percentile (99), sampler.Mean ()};
}

/* -------------------------------------------------------------------------
 * Main
 * ------------------------------------------------------------------------- */
int main (int argc, char *argv[])
{
    double mu_rate  = 1000.0;      // packets per second (service rate)
    double sim_time =   60.0;      // seconds
    uint32_t n_seeds =    10;      // seeds per rho
    std::string out_dir = "results/ns3";

    CommandLine cmd (__FILE__);
    cmd.AddValue ("muRate",  "Service rate (pkt/s)",     mu_rate);
    cmd.AddValue ("simTime", "Simulation duration (s)",  sim_time);
    cmd.AddValue ("nSeeds",  "Seeds per utilisation",    n_seeds);
    cmd.AddValue ("outDir",  "Output directory",         out_dir);
    cmd.Parse (argc, argv);

    /* Create output directory */
    std::string mkdir_cmd = "mkdir -p " + out_dir;
    std::system (mkdir_cmd.c_str ());

    std::string csv_path = out_dir + "/bottleneck_results.csv";
    std::ofstream csv (csv_path);
    csv << "rho,seed,p95_queue,p99_queue,mean_queue\n";

    std::vector<double> rhos = {0.5, 0.7, 0.8, 0.9};

    for (double rho : rhos) {
        double p95_sum = 0.0, p99_sum = 0.0;
        for (uint32_t s = 1; s <= n_seeds; ++s) {
            SimResult r = RunSim (rho, mu_rate, s * 100 + static_cast<uint32_t>(rho * 100),
                                  sim_time);
            csv << rho << "," << s << ","
                << r.p95 << "," << r.p99 << "," << r.mean << "\n";
            p95_sum += r.p95;
            p99_sum += r.p99;
            std::cout << "rho=" << rho << " seed=" << s
                      << "  P95=" << r.p95 << "  P99=" << r.p99 << "\n";
        }
        std::cout << "  => rho=" << rho
                  << "  mean_P95=" << p95_sum / n_seeds
                  << "  mean_P99=" << p99_sum / n_seeds << "\n\n";
    }

    csv.close ();
    std::cout << "Results written to: " << csv_path << "\n";
    return 0;
}
