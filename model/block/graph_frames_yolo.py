import numpy as np


COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),   # nose-eyes-ears
    (3, 5), (4, 6),                   # 直接耳連肩
    (5, 6),                           # shoulders
    (5, 7), (7, 9),                   # left arm
    (6, 8), (8, 10),                  # right arm
    (11, 5), (12, 6),                 # hips to shoulders
    (11, 12),                         # hips
    (11, 13), (13, 15),               # left leg
    (12, 14), (14, 16)                # right leg
]

class Graph():
    def __init__(self,
                 layout,
                 strategy,
                 pad=0,
                 max_hop=1,
                 dilation=1):

        self.max_hop = max_hop
        self.dilation = dilation
        self.seqlen = pad
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)

        self.dist_center = self.get_distance_to_center(layout)
        self.get_adjacency(strategy)

    def get_distance_to_center(self,layout):
        dist_center = np.zeros(self.num_node)
        if layout == 'hm36_gt':
            for i in range(self.seqlen):
                index_start = i*self.num_node_each
                dist_center[index_start+0 : index_start+7] = [1, 2, 3, 4, 2, 3, 4]
                dist_center[index_start+7 : index_start+11] = [0, 1, 2, 3]
                dist_center[index_start+11 : index_start+17] = [2, 3, 4, 2, 3, 4]
        elif layout == 'coco':
            for i in range(self.seqlen):
                index_start = i * self.num_node_each
                dc = np.full(self.num_node_each, -1)  # 預設 -1

                # hop=0 中心層(多中心)
                dc[5] = 0  # left_shoulder
                dc[6] = 0  # right_shoulder
                dc[11] = 0 # left_hip
                dc[12] = 0 # right_hip

                # hop=1
                dc[3] = 1  # left_ear
                dc[4] = 1  # right_ear
                dc[7] = 1  # left_elbow
                dc[8] = 1  # right_elbow
                dc[13] = 1 # left_knee
                dc[14] = 1 # right_knee

                # hop=2
                dc[1] = 2  # left_eye
                dc[2] = 2  # right_eye
                dc[9] = 2  # left_wrist
                dc[10] = 2 # right_wrist
                dc[15] = 2 # left_ankle
                dc[16] = 2 # right_ankle

                # hop=3
                dc[0] = 3  # nose

                dist_center[index_start : index_start + self.num_node_each] = dc
                '''
        elif layout == 'coco':
            for i in range(self.seqlen):
                index_start = i * self.num_node_each
                dc = np.full(self.num_node_each, -1)

                # hop=0 中心點（單中心）
                dc[11] = 0  # left_hip

                # hop=1
                dc[5] = 1  # left_shoulder
                dc[6] = 1  # right_shoulder
                dc[12] = 1 # right_hip
                dc[13] = 1 # left_knee

                # hop=2
                dc[3] = 2  # left_ear
                dc[4] = 2  # right_ear
                dc[7] = 2  # left_elbow
                dc[8] = 2  # right_elbow
                dc[14] = 2 # right_knee

                # hop=3
                dc[1] = 3  # left_eye
                dc[2] = 3  # right_eye
                dc[9] = 3  # left_wrist
                dc[10] = 3 # right_wrist
                dc[15] = 3 # left_ankle
                dc[16] = 3 # right_ankle

                # hop=4
                dc[0] = 4  # nose

                dist_center[index_start : index_start + self.num_node_each] = dc
                '''
        else:
            raise ValueError(f"Unsupported layout: {layout}")
        return dist_center

    def __str__(self):
        return self.A

    def graph_link_between_frames(self,base):
        return [((front - 1) + i*self.num_node_each, (back - 1)+ i*self.num_node_each) for i in range(self.seqlen) for (front, back) in base]

    def basic_layout(self,neighbour_base, sym_base):
        self.num_node = self.num_node_each * self.seqlen
        time_link = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in range(self.seqlen - 1)
                     for j in range(self.num_node_each)]
        self.time_link_forward = [(i * self.num_node_each + j, (i + 1) * self.num_node_each + j) for i in
                                  range(self.seqlen - 1)
                                  for j in range(self.num_node_each)]
        self.time_link_back = [((i + 1) * self.num_node_each + j, (i) * self.num_node_each + j) for i in
                               range(self.seqlen - 1)
                               for j in range(self.num_node_each)]

        self_link = [(i, i) for i in range(self.num_node)]

        self.neighbour_link_all = self.graph_link_between_frames(neighbour_base)

        self.sym_link_all = self.graph_link_between_frames(sym_base)

        return self_link, time_link

    def get_edge(self, layout):
        if layout == 'hm36_gt':
            self.num_node_each = 17


            neighbour_base = [(1, 2), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6),
                              (8, 1), (9, 8), (10, 9), (11, 10), (12, 9),
                              (13, 12), (14, 13), (15, 9), (16, 15), (17, 16)
                              ]
            sym_base = [(7, 4), (6, 3), (5, 2), (12, 15), (13, 16), (14, 17)]

            self_link, time_link = self.basic_layout(neighbour_base, sym_base)

            self.la, self.ra =[11, 12, 13], [14, 15, 16]
            self.ll, self.rl = [4, 5, 6], [1, 2, 3]
            self.cb = [0, 7, 8, 9, 10]
            self.part = [self.la, self.ra, self.ll, self.rl, self.cb]

            self.edge = self_link + self.neighbour_link_all + self.sym_link_all + time_link

            self.center = 8 - 1

        elif layout == 'coco':
            self.num_node_each = 17
            neighbour_base = COCO_EDGES
            sym_base = [
                    (1, 2),   # left_eye ↔ right_eye
                    (3, 4),   # left_ear ↔ right_ear
                    (5, 6),   # shoulder
                    (7, 8),   # elbow
                    (9, 10),  # wrist
                    (11, 12), # hip
                    (13, 14), # knee
                    (15, 16)  # ankle
            ]  # COCO 無明確對稱骨架資訊，可留空或自訂
            self_link, time_link = self.basic_layout(neighbour_base, sym_base)
            # 左手臂
            self.la = [5, 7, 9]    # left_shoulder → left_elbow → left_wrist
            # 右手臂
            self.ra = [6, 8, 10]   # right_shoulder → right_elbow → right_wrist
            # 左腿
            self.ll = [11, 13, 15] # left_hip → left_knee → left_ankle
            # 右腿
            self.rl = [12, 14, 16] # right_hip → right_knee → right_ankle
            # 軀幹 + 頭部中心線
            self.cb = [0, 1, 2, 3, 4, 5, 6, 11, 12] 
            # 組合成完整分組
            self.part = [self.la, self.ra, self.ll, self.rl, self.cb]

            self.edge = self_link + self.neighbour_link_all + self.sym_link_all + time_link
            self.center = 11  # COCO 中心點通常為 nose（index 0）

        else:
            raise ValueError("Do Not Exist This Layout.")





    def get_adjacency(self, strategy):

        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                a_sym = np.zeros((self.num_node, self.num_node))
                a_forward = np.zeros((self.num_node, self.num_node))
                a_back = np.zeros((self.num_node, self.num_node))
                
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.dist_center[j] == -1 or self.dist_center[i] == -1:
                                continue
                            if (j,i) in self.sym_link_all or (i,j) in self.sym_link_all:
                                a_sym[j, i] = normalize_adjacency[j, i]
                            elif (j,i) in self.time_link_forward:
                                a_forward[j, i] = normalize_adjacency[j, i]
                            elif (j,i) in self.time_link_back:
                                a_back[j, i] = normalize_adjacency[j, i]
                            elif self.dist_center[j] == self.dist_center[i]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.dist_center[j] > self.dist_center[i]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_close)
                    A.append(a_further)
                    A.append(a_sym)
                    if self.seqlen > 1:
                        A.append(a_forward)
                        A.append(a_back)

            A = np.stack(A)
            self.A = A

        else:
            raise ValueError("Do Not Exist This Strategy")
            

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


    
