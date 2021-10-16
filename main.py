import argparse
import csv
import itertools
import os
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class ArgsModel(object):
    
    def __init__(self) -> None:
        super().__init__()
    
        self.parser = argparse.ArgumentParser()
        self.parser = self.add_model_param(self.parser)


    @staticmethod
    def add_model_param(parser):
        parser.add_argument("--topo", type=str, default="square",
            help="topology of employed spatial lattices. \"square\" or \"honeycomb\"")
        parser.add_argument("--G", type=float, default=5,
            help="")
        parser.add_argument("--K", type=float, default=10**-8,
            help="default: K -> 0")
        parser.add_argument("--KG_ratio", type=float, default=0.1,
            help="K/G = KG_ratio")
        parser.add_argument("--a", type=float, default=1,
            help="")
        parser.add_argument("--l", type=float, default=800,
            help="L=400 to 1600 linear size; L=l*l; l=20 to 40")
        parser.add_argument("--r", type=float, default=5.,
            help="")
        parser.add_argument("--MCS", type=int, default=1000,
            help="Monte Carlo step (MCS)")
        parser.add_argument("--n_elestep", type=float, default=0,
            help="# of elementary steps")
        parser.add_argument("--seed", type=int, default=11,
            help="")
        return parser
    

    @staticmethod
    def post_processing(args):
        args.n_elestep = args.l*args.l
        args.K = args.G * args.KG_ratio
        return args


    def get_args(self):
        args = self.parser.parse_args()
        return self.post_processing(args)


class Agent(object):
    _ids = itertools.count(0)

    def __init__(self, args) -> None:
        super().__init__()
        
        self.id = next(self._ids)
        self.isCooperator = self.draw(p=0.5)
        self.net = None
        self.payoff = None

        self.a = args.a
        self.r = args.r
    

    @staticmethod
    def draw(p):
        return True if np.random.uniform() < p else False


    def _get_net_gain(self):
        if self.net is None:
            raise TypeError("net is not initialized.")
        payoff = self.a*len([ag for ag in self.net if ag.isCooperator])*self.r / len(self.net)
        return payoff
    
    def get_payoff(self):
        payoff = sum([ag._get_net_gain() for ag in self.net])
        if self.isCooperator:
            payoff -= self.a * len(self.net)
        return payoff


class UnbalancedEdgeHolder(object):
    def __init__(self) -> None:
        super().__init__()
        self.s = set()
    
    def add(self, i, j):
        if not self.check(i, j):
            self.s.add(self.to_key(i, j))
    
    def check(self, i, j):
        return self.to_key(i, j) in self.s
    
    def pop(self):
        i, j = self.s.pop().split("_")
        return int(i), int(j)
    
    def remove(self, i, j):
        if self.check(i, j):
            self.s.remove(self.to_key(i, j))
    
    def isempty(self):
        return not bool(self.s)
    
    @staticmethod
    def to_key(i, j):
        if i > j:
            i, j = j, i
        return "{}_{}".format(i, j)
    

class PublicGoodsGame(object):
    adjacent = [(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)]

    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        print("Args: {}".format(args))
        
        self.ags, self.n_ags, self.u_edge_set, self.id2ag = self.init_ag()
        self.n_c = self.get_n_c()
        print("init {} agents: {} cooperators, {} defectors; rho_c: {:.4f}".format(self.n_ags, 
            self.n_c, self.n_ags-self.n_c, self.n_c/self.n_ags))
        
        self.rho_c = list()
    

    def set_r(self, r):
        self.args.r = r
    

    def init_ag(self):
        # init agents
        if self.args.topo == "square":
            ags = [[Agent(self.args) for _ in range(self.args.l)] for _ in range(self.args.l)]
            n_ags = self.args.l * self.args.l
            u_edge_set = UnbalancedEdgeHolder()
            id2ag = dict()
        else:
            raise TypeError("topo other than square is not supported.")
        
        # build net
        if self.args.G == 5:
            for i in range(self.args.l):
                for j in range(self.args.l):
                    id2ag[ags[i][j].id] = ags[i][j]
                    ags[i][j].net = [ags[(i+di)%self.args.l][(j+dj)%self.args.l] for di, dj in PublicGoodsGame.adjacent]
                    
                    # check unbalanced edges
                    for ag in ags[i][j].net:
                        if ags[i][j].isCooperator != ag.isCooperator:
                            u_edge_set.add(ags[i][j].id, ag.id)
        else:
            raise TypeError("G other than 5 is not supported.")
        
        return ags, n_ags, u_edge_set, id2ag
    
    
    def get_n_c(self):
        return sum([1 for ag_ls in self.ags for ag in ag_ls if ag.isCooperator])


    def simulate_step(self):
        """
        1. choose a agent x, then one of its neighbors y
        2. enforce x's strategy s_x on player y based on a prob
        """

        if self.u_edge_set.isempty():
            return True
        
        # ag_x_id, ag_y_id = self.u_edge_set.pop()
        # ag_x = self.id2ag[ag_x_id]
        # ag_y = self.id2ag[ag_y_id]
        ag_x = self.ags[np.random.randint(self.args.l)][np.random.randint(self.args.l)]
        ag_y = ag_x.net[np.random.randint(len(ag_x.net))]
        if ag_x.isCooperator == ag_y.isCooperator:
           return

        prob = 1 / (1 + np.exp((ag_y.get_payoff()-ag_x.get_payoff())/self.args.K))
        if np.random.uniform() < prob:
            ag_y.isCooperator = ag_x.isCooperator
            # update y
            for ag in ag_y.net:
                if ag.isCooperator == ag_y.isCooperator:
                    self.u_edge_set.remove(ag_y.id, ag.id)
                else:
                    self.u_edge_set.add(ag_y.id, ag.id)
        return False
    

    def simulate(self, log_v=1):
        terminate = False
        for mcs_str in range(self.args.MCS):
            for s_ctr in range(self.args.n_elestep):
                terminate = self.simulate_step()
                if terminate:
                    break
            n_c = self.get_n_c()
            print("MCS {}: rho_c {:.4f}".format(mcs_str, n_c/self.n_ags))
            self.rho_c.append(n_c/self.n_ags)
            if terminate:
                break


def play_game(args, rg_ratio):
    print("Running on the process {}".format(os.getpid()))
    game = PublicGoodsGame(args)
    game.set_r(args.G * rg_ratio)
    game.simulate()

    log_info_file = open(log_info_fn, 'a', newline='')
    log_info_writer = csv.writer(log_info_file)
    log_info_writer.writerow([rg_ratio, game.rho_c[-1]]+game.rho_c)
    log_info_file.close()


class PlotLinesHandler(object):

    def __init__(self, xlabel="MCS", ylabel="rho_c", ylabel_show=r"$\rho_c$", title_param={},
        figure_size=(9, 9), output_dir=os.path.join(os.getcwd(), "imgfiles")) -> None:
        super().__init__()

        self.output_dir = output_dir
        self.title = "{}-{}".format(ylabel, xlabel)
        self.title_param = "_".join(["{}-{}".format(p_n, p_val) for p_n, p_val in title_param.items()])
        self.tried_param = list()

        plt.figure(figsize=figure_size, dpi=80)
        plt.title("{}-{}".format(ylabel_show, xlabel))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel_show)

    def plot_line(self, data, param_val, linewidth=1):
        self.tried_param.append("{:.2f}".format(float(param_val)))
        plt.plot(np.arange(data.shape[-1]), data, linewidth=linewidth)

    def save_fig(self, param_n="r"):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        plt.legend(["{}={}".format(param_n, p) for p in self.tried_param])
        title_tried_param = "{}-{}".format(param_n, "-".join(self.tried_param))
        fn = "_".join([self.title, self.title_param, title_tried_param]) + ".png"
        plt.savefig(os.path.join(self.output_dir, fn))
        print("fig save to {}".format(os.path.join(self.output_dir, fn)))


def plot_rho_c_rG_ratio(log_info_fn, xlabel="enhancement factor r/G", ylabel="rho_c", ylabel_show=r"$\rho_c$",
    figure_size=(6, 6), output_dir=os.path.join(os.getcwd(), "imgfiles")):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=figure_size, dpi=80)
    plt.title("rho_c-r/G".format(ylabel_show, xlabel))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel_show)
    plt.ylim(0.0, 1.0)

    rg_ratio_ls, final_rho_c_ls = list(), list()
    with open(log_info_fn, newline="") as log_info_file:
        log_info_reader = csv.reader(log_info_file)
        for row_idx, row in enumerate(log_info_reader):
            if row_idx == 0:
                continue
            rg_ratio_ls.append(float(row[0]))
            final_rho_c_ls.append(float(row[1]))
    
    plt.scatter(rg_ratio_ls, final_rho_c_ls)

    fn = "_".join(["rho_c-rG"] + os.path.splitext(log_info_fn)[0].split("_")[-3:] + ["rG-"+"-".join(["{:.2f}".format(rg) for rg in rg_ratio_ls])]) + ".png"
    plt.savefig(os.path.join(output_dir, fn))
    print("fig save to {}".format(os.path.join(output_dir, fn)))




if __name__ == "__main__":
    parser = ArgsModel()
    args = parser.get_args()
    np.random.seed(args.seed)

    log_info_fn = "log_info_G-{}_K-{}_L-{}.csv".format(args.G, args.K, args.l*args.l)
    if not os.path.exists(log_info_fn):
        log_info_file = open(log_info_fn, 'w', newline='')
        log_info_writer = csv.writer(log_info_file)
        log_info_writer.writerow(["r/G", "final rho_c"] + ["MCS "+str(i) for i in range(args.MCS)])
        log_info_file.close()
    
    n_cpus = multiprocessing.cpu_count()
    args_play_games = [(args, rg_ratio) for rg_ratio in np.arange(0.70, 1.21, 0.02)]
    print("cpu count: {}".format(n_cpus))
    pool = multiprocessing.Pool(n_cpus)
    pool.starmap(play_game, args_play_games)
    
    plot_rho_c_rG_ratio(log_info_fn)

    #plot_line_handler = PlotLinesHandler(xlabel="MCS", ylabel="rho_c", title_param={"L": args.l*args.l})
    #plot_line_handler.plot_line(data=np.array(game.rho_c), param_val=rg_ratio)
    #plot_line_handler.save_fig(param_n="rG")