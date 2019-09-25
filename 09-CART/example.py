from numpy import *
from matplotlib.font_manager import FontProperties
from tkinter import *
import matplotlib

# 将matplotlib后端设置为TkAgg
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.tree import DecisionTreeRegressor


def load_data_set(file_name):
    """
        Function:
            加载数据
        Parameters:
            file_name - 文件名
        Returns:
            data_mat - 数据矩阵
    """
    data_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = list(map(float, cur_line))
        data_mat.append(flt_line)
    return data_mat


# 根据特征切分数据集合
def bin_split_data_set(data_set, feature, value):
    """
        Function:
            根据特征切分数据集合
        Parameters:
            data_set - 数据集合
            feature - 待切分的特征
            value - 该特征的值
        Returns:
            mat0 - 切分的数据集合0
            mat1 - 切分的数据集合1
    """
    mat0 = data_set[nonzero(data_set[:, feature] > value)[0], :]
    mat1 = data_set[nonzero(data_set[:, feature] <= value)[0], :]
    return mat0, mat1


def reg_leaf(data_set):
    """
        Function:
            生成叶结点
        Parameters:
            data_set - 数据集合
        Returns:
            mean - 目标变量的均值
    """
    return mean(data_set[:, -1])


def reg_err(data_set):
    """
        Function:
            生成叶结点
        Parameters:
            data_set - 数据集合
        Returns:
            var - 目标变量的总方差
    """
    # var()计算平方误差
    return var(data_set[:, -1]) * shape(data_set)[0]


def choose_best_split(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    """
        Function:
            找到数据的最佳二元切分方式函数
        Parameters:
            data_set - 数据集合
            leaf_type - 生成叶结点
            err_type - 误差估计函数
            ops - 用户定义的参数构成的元组
        Returns:
            best_index - 最佳切分特征
            best_value - 最佳特征值
    """
    # tol_s允许的误差下降值，tol_n切分的最少样本数
    tol_s = ops[0]
    tol_n = ops[1]
    # 将数组或者矩阵转换成列表，统计不同剩余特征值的数目
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(data_set)
    m, n = shape(data_set)
    s = err_type(data_set)
    # 分别为最佳误差，最佳特征切分的索引值，最佳特征值
    best_s = float('inf')
    best_index = 0
    best_value = 0
    # 遍历所有特征列
    for feat_index in range(n - 1):
        # 遍历所有特征值
        for split_val in set(data_set[:, feat_index].T.A.tolist()[0]):
            # 根据特征和特征值切分数据集
            mat0, mat1 = bin_split_data_set(data_set, feat_index, split_val)
            # 如果数据少于tol_n,则退出
            if (shape(mat0)[0] < tol_n) or (shape(mat1)[0] < tol_n): continue
            # 计算误差估计
            new_s = err_type(mat0) + err_type(mat1)
            # 如果误差估计更小,则更新最佳特征索引值和特征值
            if new_s < best_s:
                best_index = feat_index
                best_value = split_val
                best_s = new_s
    # 如果误差减少不大则退出
    if (s - best_s) < tol_s:
        return None, leaf_type(data_set)
    # 根据最佳的切分特征和特征值切分数据集合
    mat0, mat1 = bin_split_data_set(data_set, best_index, best_value)
    # 如果切分出的数据集很小则退出
    if (shape(mat0)[0] < tol_n) or (shape(mat1)[0] < tol_n):
        return None, leaf_type(data_set)
    # 返回最佳切分特征和特征值
    return best_index, best_value


def create_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    """
        Function:
            构建回归树
        Parameters:
            data_set - 数据集合
            leaf_type - 建立叶结点的函数
            err_type - 误差计算函数
            ops - 包含树构建所有其他参数的元组
        Returns:
            ret_tree - 构建的回归树
    """
    feat, val = choose_best_split(data_set, leaf_type, err_type, ops)
    if feat == None: return val
    ret_tree = {}
    ret_tree['spInd'] = feat
    ret_tree['spVal'] = val
    lset, rset = bin_split_data_set(data_set, feat, val)
    # print(lset)
    # print(rset)
    ret_tree['left'] = create_tree(lset, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(rset, leaf_type, err_type, ops)
    return ret_tree


def plot_data_set(file_name):
    """
        Function:
            绘制数据集
        Parameters:
            file_name - 文件名
        Returns:
            无
    """
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    data_mat = load_data_set(file_name)
    n = len(data_mat)   #数据个数
    xcord = []  #样本点列表初始化
    ycord = []
    for i in range(n):
        # ex00.txt、ex2.txt、exp2.txt数据集
        xcord.append(data_mat[i][0])	#加入样本点
        ycord.append(data_mat[i][1])
        # ex0.txt数据集
        # xcord.append(data_mat[i][1])
        # ycord.append(data_mat[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=0.5)	#绘制样本点
    # plt.title('data_set')
    # plt.xlabel('X')
    ax_xlabel_text = ax.set_xlabel(u'骑自行车的速度', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'智商（IQ）', FontProperties=font)
    plt.show()


def is_tree(obj):
    """
        Function:
            判断测试输入变量是否是一棵树
        Parameters:
            obj - 判断对象
        Returns:
            是否是一棵树
    """
    return (type(obj).__name__ == 'dict')


def get_mean(tree):
    """
        Function:
            对树进行塌陷处理(即返回树平均值)
        Parameters:
            tree - 树
        Returns:
            树的平均值
    """
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, test_data):
    """
        Function:
            后剪枝
        Parameters:
            tree - 树
            test_data - 测试集
        Returns:
            后剪枝的树
    """
    # 如果测试集为空，则对树进行塌陷处理
    if shape(test_data)[0] == 0:
        return get_mean(tree)
    # 如果有左子树或者右子树，则切分数据集
    if (is_tree(tree['right']) or is_tree(tree['left'])):
        lset, rset = bin_split_data_set(test_data, tree['spInd'], tree['spVal'])
    # 处理左子树(剪枝)
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], lset)
    # 处理右子树(剪枝)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], rset)
    # 如果当前结点的左右结点为叶结点
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        lset, rset = bin_split_data_set(test_data, tree['spInd'], tree['spVal'])
        # 计算没有合并的误差
        error_nomerge = sum(power(lset[: -1] - tree['left'], 2)) + sum(power(rset[:, -1] - tree['right'], 2))
        # 计算合并的均值
        tree_mean = (tree['left'] + tree['right']) / 2.0
        # 计算合并的误差
        error_merge = sum(power(test_data[:, -1] - tree_mean, 2))
        # 如果合并的误差小于没有合并的误差，则合并
        if error_merge < error_nomerge:
            return tree_mean
        else:
            return tree
    else:
        return tree


def liner_solve(data_set):
    """
        Function:
            模型树叶节点生成
        Parameters:
            data_set - 数据集
        Returns:
            ws -回归系数
            x - 数据集矩阵
            y - 目标变量值矩阵
    """
    m, n = shape(data_set)
    # 构建大小为(m,n)和(m,1)的矩阵
    x = mat(ones((m, n)))
    y = mat(ones((m, 1)))
    # 数据集矩阵的第一列初始化为1，偏置项
    x[:, 1:n] = data_set[:, 0:n - 1]
    # 每个样本目标变量值存入Y
    y = data_set[:, -1]
    # 对数据集矩阵求内积
    x_t_x = x.T * x
    # 计算行列式值是否为0，即判断是否可逆
    if linalg.det(x_t_x) == 0.0:
        # 不可逆，打印信息
        raise NameError('This matrix is singular, cannot du inverse, \ntry increasing the second value of ops')
    # 可逆，计算回归系数
    ws = x_t_x.I * (x.T * y)
    return ws, x, y


def model_leaf(data_set):
    """
        Function:
            模型树的叶节点模型，当数据不再需要切分时生成叶节点的模型
        Parameters:
            data_set - 数据集
        Returns:
            ws -回归系数
    """
    ws, x, y = liner_solve(data_set)
    return ws


def model_err(data_set):
    """
        Function:
            模型树的误差计算函数
        Parameters:
            data_set - 数据集
        Returns:
            返回误差平方和，平方损失
    """
    ws, x, y = liner_solve(data_set)
    y_hat = x * ws
    return sum(power(y - y_hat, 2))


def plot_model_data_set(data_set, tree):
    """
        Function:
            绘制模型树
        Parameters:
            data_set - 数据集
            tree - 模型树
        Returns:
            无
    """
    n = len(data_set)
    xcord = []
    ycord = []
    xcord_1 = []
    xcord_2 = []
    for i in range(n):
        xcord.append(data_set[i][0])
        ycord.append(data_set[i][1])
        if data_set[i][0] <= tree['spVal']:
            xcord_1.append(data_set[i][0])
        else:
            xcord_2.append(data_set[i][0])
    xcord_1_mat = mat(xcord_1)
    xcord_2_mat = mat(xcord_2)
    ws_1 = tree['right']
    ws_2 = tree['left']
    y_hat_1 = ws_1[0] + ws_1[1] * xcord_1_mat
    y_hat_2 = ws_2[0] + ws_2[1] * xcord_2_mat
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xcord_1_mat.flatten().A[0], y_hat_1.flatten().A[0], c='red')
    ax.plot(xcord_2_mat.flatten().A[0], y_hat_2.flatten().A[0], c='red')
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=0.5)
    plt.title('data_set')
    plt.xlabel('X')
    plt.show()


def reg_tree_eval(model, in_dat):
    """
        Function:
            用树回归进行预测代码
        Parameters:
            model - 树回归模型
            in_dat - 输入数据
        Returns:
            回归树的叶节点为float型常量
    """
    return float(model)


def model_tree_eval(model, in_dat):
    """
        Function:
            模型树的叶节点浮点型参数的线性方程
        Parameters:
            model - 模型树模型
            in_dat - 输入数据
        Returns:
            浮点型的回归系数向量
    """
    # 获取输入数据的列数
    n = shape(in_dat)[1]
    # 构建n+1维的单列矩阵
    x = mat(ones((1, n + 1)))
    # 第一列设置为1，线性方程偏置项b
    x[:, 1:n + 1] = in_dat
    return float(x * model)


def tree_fore_cast(tree, in_data, model_eval=reg_tree_eval):
    """
        Function:
            树预测
        Parameters:
            tree - 树回归模型
            in_dat - 输入数据
            model - 模型树模型
        Returns:
            浮点型的回归系数向量
    """
    # 如果当前树为叶节点，生成叶节点
    if not is_tree(tree): return model_eval(tree, in_data)
    # 非叶节点，对该子树对应的切分点对输入数据进行切分
    if in_data[tree['spInd']] > tree['spVal']:
        # 该树的左分支为非叶节点类型
        if is_tree(tree['left']):
            # 递归调用treeForeCast函数继续树预测过程，直至找到叶节点
            return tree_fore_cast(tree['left'], in_data, model_eval)
        else:
            # 左分支为叶节点，生成叶节点
            return model_eval(tree['left'], in_data)
    # 小于切分点值的右分支
    else:
        # 非叶节点类型
        if is_tree(tree['right']):
            # 非叶节点类型
            return tree_fore_cast(tree['right'], in_data, model_eval)
        else:
            # 叶节点，生成叶节点类型
            return model_eval(tree['right'], in_data)


def create_fore_cast(tree, test_data, model_eval=reg_tree_eval):
    """
        Function:
            创建预测树
        Parameters:
            tree - 树回归模型
            test_data - 测试数据
            model - 模型树模型
        Returns:
            预测值
    """
    m = len(test_data)
    # 初始化行向量各维度值为1
    y_hat = mat(zeros((m, 1)))
    for i in range(m):
        y_hat[i, 0] = tree_fore_cast(tree, mat(test_data[i]), model_eval)
    return y_hat


def re_draw(tol_s, tol_n):
    """
        Function:
            以用户输入的终止条件为参数绘制树
        Parameters:
            tol_s - 允许的误差下降值
            tol_n - 切分的最少样本数
        Returns:
            无
    """
    # 清空之前的图像
    re_draw.f.clf()
    re_draw.a = re_draw.f.add_subplot(111)
    # 检查复选框是否选中，选中就构建模型树，选中默认构回归树
    if chk_btn_val.get():
        if tol_n < 2: tol_n = 2
        my_tree = create_tree(re_draw.raw_dat, model_leaf, model_err, (tol_s, tol_n))
        y_hat = create_fore_cast(my_tree, re_draw.test_dat, model_tree_eval)
    else:
        my_tree = create_tree(re_draw.raw_dat, ops=(tol_s, tol_n))
        y_hat = create_fore_cast(my_tree, re_draw.test_dat)
    # 绘制真实值
    re_draw.a.scatter(re_draw.raw_dat[:, 0].tolist(), re_draw.raw_dat[:, 1].tolist(), s=5)
    # 绘制预测值
    re_draw.a.plot(re_draw.test_dat, y_hat, linewidth=2.0)
    re_draw.canvas.show()


def get_inputs():
    """
        Function:
            从文本输入框中获取树创建终止条件，没有则用默认值
        Parameters:
            无
        Returns:
            tol_s - 允许的误差下降值
            tol_n - 切分的最少样本数
    """
    try:
        tol_n = int(tol_n_entry.get())
    # 清除错误的输入并用默认值替换
    except:
        tol_n = 10
        print('enter Integer for tol_n')
        tol_n_entry.delete(0, END)
        tol_n_entry.insert(0, '10')
    try:
        tol_s = float(tol_n_entry.get())
    except:
        tol_n = 10
        print('enter Integer for tol_s')
        tol_n_entry.delete(0, END)
        tol_n_entry.insert(0, '1.0')
    return tol_n, tol_s


def draw_new_tree():
    """
        Function:
            按下re_draw按钮时，开始绘图
        Parameters:
            无
        Returns:
            无
    """
    # 从文本输入框中获取树创建终止条件
    tol_n, tol_s = get_inputs()
    re_draw(tol_s, tol_n)


if __name__ == '__main__':
    # test_mat = mat(eye(4))
    # mat0, mat1 = bin_split_data_set(test_mat, 1, 0.5)
    # print('原始集合:\n', test_mat)
    # print('mat0:\n', mat0)
    # print('mat1:\n', mat1)

    # file_name = './machinelearninginaction/Ch09/ex00.txt'
    # plot_data_set(file_name)

    # data_set = load_data_set('./machinelearninginaction/Ch09/ex00.txt')
    # data_mat = mat(data_set)
    # best_feat, val = choose_best_split(data_mat)
    # print(best_feat)
    # print(val)


    # data_set = load_data_set('./machinelearninginaction/Ch09/ex00.txt')
    # data_mat = mat(data_set)
    # my_tree = create_tree(data_mat)
    # print(my_tree)

    # file_name = './machinelearninginaction/Ch09/ex0.txt'
    # plot_data_set(file_name)

    # data_set = load_data_set('./machinelearninginaction/Ch09/ex0.txt')
    # data_mat = mat(data_set)
    # my_tree = create_tree(data_mat)
    # print(my_tree)


    # 预剪枝
    # file_name = './machinelearninginaction/Ch09/ex2.txt'
    # plot_data_set(file_name)

    # data_set = load_data_set('./machinelearninginaction/Ch09/ex2.txt')
    # data_mat = mat(data_set)
    # my_tree = create_tree(data_mat, ops=(1000, 4))
    # print(my_tree)

    # 后剪枝
    # print('剪枝前：')
    # train_data = load_data_set('./machinelearninginaction/Ch09/ex2.txt')
    # train_mat = mat(train_data)
    # tree = create_tree(train_mat)
    # print(tree)
    # print('\n剪枝后：')
    # test_data = load_data_set('./machinelearninginaction/Ch09/ex2test.txt')
    # test_mat = mat(test_data)
    # print(prune(tree, test_mat))


    # file_name = './machinelearninginaction/Ch09/exp2.txt'
    # plot_data_set(file_name)

    # data_set = load_data_set('./machinelearninginaction/Ch09/exp2.txt')
    # data_mat = mat(data_set)
    # tree = create_tree(data_mat, model_leaf, model_err, (1, 10))
    # print(tree)

    # data_set = load_data_set('./machinelearninginaction/Ch09/exp2.txt')
    # data_mat = mat(data_set)
    # tree = create_tree(data_mat, model_leaf, model_err, (1, 10))
    # plot_model_data_set(data_set, tree)


    # plot_data_set('./machinelearninginaction/Ch09/bikeSpeedVsIq_train.txt')


    # train_mat = mat(load_data_set('./machinelearninginaction/Ch09/bikeSpeedVsIq_train.txt'))
    # test_mat = mat(load_data_set('./machinelearninginaction/Ch09/bikeSpeedVsIq_test.txt'))
    # # 回归树
    # reg_tree = create_tree(train_mat, ops=(1, 20))
    # # 模型树
    # model_tree = create_tree(train_mat, model_leaf, model_err, ops=(1, 20))
    # y_hat_reg_tree = create_fore_cast(reg_tree, test_mat[:, 0])
    # y_hat_model_tree = create_fore_cast(model_tree, test_mat[:, 0], model_tree_eval)
    # relation_reg_tree = corrcoef(y_hat_reg_tree, test_mat[:, 1], rowvar=0)[0, 1]
    # relation_model_tree = corrcoef(y_hat_model_tree, test_mat[:, 1], rowvar=0)[0, 1]
    # # 线性方程
    # ws, x, y = liner_solve(train_mat)
    # m = shape(test_mat)[0]
    # y_hat = mat(zeros((m, 1)))
    # for i in range(m):
    #     y_hat[i] = test_mat[i, 0] * ws[1, 0] + ws[0, 0]
    # relation_linear_solve = corrcoef(y_hat, test_mat[:, 1], rowvar=0)[0, 1]
    # print('回归树相关系数:', relation_reg_tree)
    # print('模型树相关系数:', relation_model_tree)
    # print('线性方程相关系数:', relation_linear_solve)

    # root = Tk()
    #
    # # 标签部件
    # # Label(root, text='Plot Place Holder').grid(row=0, columnspan=3)
    # re_draw.f = Figure(figsize=(5, 4), dpi=100)
    # re_draw.canvas = FigureCanvasTkAgg(re_draw.f, master=root)
    # re_draw.canvas.show()
    # re_draw.canvas.get_tk_widget().grid(row=0, columnspan=3)
    # # 文本输入框部件tol_n
    # Label(root, text='tol_n').grid(row=1, column=0)
    # tol_n_entry = Entry(root)
    # tol_n_entry.grid(row=1, column=0)
    # tol_n_entry.insert(0, '10')
    # # 文本输入框部件
    # Label(root, text='tol_s').grid(row=2, column=0)
    # tol_n_entry = Entry(root)
    # tol_n_entry.grid(row=2, column=0)
    # tol_n_entry.insert(0, '1.0')
    # # 按钮部件
    # Button(root, text='re_draw', comman=draw_new_tree).grid(row=1, column=2, rowspan=3)
    # chk_btn_val = IntVar()
    # chk_btn = Checkbutton(root, text='Model Tree', variable=chk_btn_val)
    # chk_btn.grid(row=3, column=0, columnspan=2)
    # re_draw.raw_dat = mat(load_data_set('./machinelearninginaction/Ch09/sine.txt'))
    # re_draw.test_dat = arange(min(re_draw.raw_dat[:, 0]), max(re_draw.raw_dat[:, 0]), 0.01)
    #
    # re_draw(1.0, 10)
    # root.mainloop()

    train_mat = mat(load_data_set('./machinelearninginaction/Ch09/bikeSpeedVsIq_train.txt'))
    test_mat = mat(load_data_set('./machinelearninginaction/Ch09/bikeSpeedVsIq_test.txt'))
    reg_tree = DecisionTreeRegressor(max_depth=4)
    reg_tree = reg_tree.fit(train_mat[:, 0], train_mat[:, 1])
    y_hat_reg_tree = reg_tree.predict(test_mat[:, 0])
    relation_reg_tree = corrcoef(y_hat_reg_tree, test_mat[:, 1], rowvar=0)[0, 1]
    print('模型树相关系数:', relation_reg_tree)
