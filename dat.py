'''

ps:#a=[i*i for i in range(5) if i<3 ]  #python for if 的一行写法.
https://segmentfault.com/a/1190000008877595#articleHeader7
5.4.2 Base Array 的构造
看这里面写的还真不难,之前一直没看懂,是因为他数据没有显示写入.
其实还有一个数组用来写入数据.比如
这里面第一步之后的data数组变成了
data[2]='清'
data[3]='华'
data[7]='中'
这样通过他的步骤,做到最后就是3个数组,data,base,check3个数组来
表示这个2array trie.就能方便找到每一个词组了.
但是写起来简直吐血.

首先看最终得到的结果如何使用它来找到所有的词组:

字典:''清华”、“清华大学”、“清新”、“中华”、“华人”
编码:清-1，华-2，大-3，学-4，新-5，中-6，人-7

数组下表:0    1   2   3   4   5   6   7   8   9   10
base数组:1    空  3   2   2   3   6   2   3   2    6

1.使用:找清华:首先从base[0]出发.清在的位置是base[0]+code(清)=下表为2的地方
           清的base数组不是负数,说明有继续拓展的本事.所以找下一个词华可以找.
           华=他上一个节点的base值+code(华)=3+2=5.所以就找到了清华在我们字典里面存在
       找清华大学:上面华找到了,继续找大=base(华)+code(大)=5(注意是清华的华,所以是上面找到的3)+3=6
                  继续找学=base[6]+code(学)=10.所以清华大学找到了.
  继续细化:叶子节点的处理:将词的最后一个节点的转移基数统一改为某个负数
           所以 数组下表:0    1   2   3   4   5    6    7    8    9   10
                base数组:1    空  3   2   -2   -3   6   2   -3   -2    -6
          这样做的代价就是需要将状态转移函数base[s]+code(字符)改为|base[s]|+code(字符)
          重新跑一次清华:上来还是清=1+1=2   华=3+2=5  然后看base[5]=-3 ,所以可以到此结束来组成一个词汇.
          但是我们还可以继续跑
          来找清华大学:从华找大:大=|-3|+code(大)=6,base[6]不是负数,不能输出.
                      继续找学:学=6+4=10,他的base是-6.所以可以输出.
  加入check数组来解决bug:比如找'清中':找清我们到了3,找中我们到了9.base[9]=-2.所以我们输出'清中'是一个词汇.
                        这显然是错误的!所以我们要加入check数组来避免这种匹配.这种bug的原因是中这个词前面
                        不能是清这个字.用check数组来记录这个位置前面一个字符所在的index.
          所以 数组下表:0    1   2   3   4   5    6    7    8    9   10
               base数组:1    空  3   2   -2   -3   6   2   -3   -2    -6
               check  :-3   -1   0   0   7   2     5   0   2    3     6
               这样找清中:清是到了index2.判断check是不是清的上一个节点.是0(0表示根)没问题.
                         找中找到index9.然后需要判断check[9]是不是他过来的节点的index.发现一个是2,一个是3
                         所以不对.输出清中不存在.
2.搭建:
https://blog.csdn.net/kissmile/article/details/47417277
这个写的也是不错.但是他搭建的顺序有一点错误,按照层搭建,第五部分应该是搭建第一层的b后面的c节点.
逻辑基本就是这样,能讲清楚就不错了.基本达到智商110以上了.能代码实现感觉智商上150了.
因为比较复杂,还是先写伪代码.再实现.


题目:建立字典:字典:''清华”、“清华大学”、“清新”、“中华”、“华人”
伪代码过程:
●a=[''清华”、“清华大学”、“清新”、“中华”、“华人”],b=sum([len(i) for i in a])
●对set(a)进行编码:清-1，华-2，大-3，学-4，新-5，中-6，人-7
●建立首字集合c:清,中,华
●为了数组足够长,建立base=[0]*b  check=[0]*b
●把c插入双数组,对base[0]赋予初值1.(其实赋予2也一样,貌似更好,因为初值1基本都会发生冲突,会降低建立速度)
 对新建立的base里面也放入1.
 把c插入后:
 数组下表:0    1   2   3   4   5   6   7   8   9   10
 base数组:1    0   1   1   0   0   0   1    0   0    0
 check  :0    0   0   0   0   0   0   0    0   0    0

●下面就是插入第二个字:华,新,华,人(第一个华,表示清后面的华,虽然他有2个但是前面都是清,所以只插入一个,这就是为什么
 Trie树节省空间的原因).
 下面插入清后面的字:有华和新(对于同一个字的后面的字要一起考虑,因为可能要修改这同一个的base数组)
 从2开始跑,华=base[2]+code(华)=3.冲突了因为3里面已经有了.
 所以base[2]+=1.这时再算华=4了.不冲突了.
 插入新又冲突了.所以清要继续加1.插入后的新元素base还是置1.(但是网上写的是置清现在的base值.我感觉没必要啊!!!!)
 也就是下图5,8我都置1,但是网上置的是3.(下面通过我的计算,我置1最后还是为了解决冲突而加到3了.
 难道置3能减少冲突的发生?问题是会不会空间浪费太多?)(利用树来看就是树的第n层的偏移量一定比第n-1层的至少一样或者多)
 (为什么?)(我认为是从概率上来讲,每一个字符边上的字符数量都一样,所以你上个字母需要偏移3个才能不冲突,
 你也至少需要偏移3个.减少代码运行时间.要知道处理冲突非常非常慢!!!!!)
 同时把check也更新了,也就是把清的index 2放进去.
 得到:

 数组下表:0    1   2   3   4   5   6   7   8   9   10
 base数组:1    0   3   1   0   1   0   1    1   0    0
 check  : 0    0   0   0   0   2   0   0    2   0    0


 (!!!!!!这里面就是遇到一个问题非常重要.搭建时候一定要多行一起搭建,也就是按照root的一层来搭建.把一层都弄好
 再弄下一层,原因就是我们最后需要得到的树是一个公共前缀只保存一次的树!也是问题的根本,不保持的话这个trie树
 完全没意义了,所以公共前缀保持同时处理,所以只能这样按照root的层来搭建才可以.)
 同理插入中后面的字:7的base+=1.得到:
 数组下表:0    1   2   3   4   5   6   7   8   9   10
 base数组:1    0   3   1   1   1   0   2    1   0    0
 check  : 0    0   0   0   7   2   0   0    2   0    0

 同理华人:得到:
 数组下表:0    1   2   3   4   5   6   7   8   9   10
 base数组:1    0   3   2   1   1   0   2    1   1    0
 check  : 0    0   0   0   7   2   0   0    2   3    0


 第三层.
 得到:
 数组下表:0    1   2   3   4   5   6   7   8   9   10
 base数组:1    0   3   2   1   3   1   2    1   1    0
 check  : 0    0   0   0   7   2   5   0    2   3    0

  第四层.
 得到:
 数组下表:0    1   2   3   4   5   6   7   8   9   10
 base数组:1    0   3   2   1   3   6   2    1   1    1
 check  : 0    0   0   0   7   2   5   0    2   3    6



 总结:难度不比红黑树简单.
'''


class DAT():
    def __init__(self, data):  # 通过这个函数返回self.base和self.check 2个数组
        # 对data预处理:
        firststep = []
        max_ceng = 0  # 数据有多少层
        for i in data:
            a = 0
            for j in i:
                firststep.append(j)
                a += 1
            if a > max_ceng:
                max_ceng = a
        all_len = len(firststep)
        mono_len = len(set(firststep))

        # 用字典进行编码.用数组太慢了,因为数组里面搜索是O(N)
        bianma = {}
        ma = 1
        tmp = []
        for i in firststep:  # 这里面去重,为了测试先这么写保顺序,写好后再改用set来加速
            if i not in tmp:
                tmp.append(i)
        for i in tmp:
            if i not in bianma:
                bianma[i] = ma
                ma += 1
        # 我为了方便把''作为root,给他bianma 是0,然后base[0]=1
        bianma[''] = 0  # 只是为了递归写起来代码更简洁而已.自我感觉很简约.
        # 初始化base 和check
        base = ['#'] * all_len  # 虽然相同也不要用等号给check赋值base,因为list赋值是浅拷贝,传的是地址
        base[0] = 1
        check = ['#'] * all_len
        # 打印一下编码看看,因为字典是乱序的,每一次生成都不同,所以打印一下来验算自己做的对不对.
        print(bianma)
        self.bianma = bianma
        # 开始建立:
        # 建立是按照第一列,...,最后一列这个顺序进行递归的.
        # 提取当前列的set后元素.
        # 第一列可以看做''空字符开始的后面一个元素.
        # 提取第一列:然后再递归修改成提取第i列

        before = ''
        col_now = [i[len(before)] for i in data if before in i]  # 提取有before前缀的字符的下一个小字符.#第一层就是清,华,中
        tmp = []
        for i in col_now:
            if i not in tmp:
                tmp.append(i)
        col_now = tmp
        print('第一列')
        print(col_now)
        # 开始计算col_now里面的字符的base
        before_index = bianma[before]  # 其他层不是这么算的.
        now_layer_save_for_data = []  # 为了下一层的递推而记录的文字信息
        now_layer_save_for_base = []  # 为了下一层的递推而记录的index信息
        for i in col_now:

            while 1:
                index = base[before_index] + bianma[i]
                if base[index] == '#':  # 说明没有人占用
                    base[index] = base[before_index]
                    check[index] = before_index
                    now_layer_save_for_data.append(i)
                    now_layer_save_for_base.append(index)
                    break
                else:
                    base[before_index] += 1
        last_layer = 1
        print('第一层')
        print(base)  # 测试后第一层建立成功.
        print(check)
        print(max_ceng)
        print(now_layer_save_for_data)
        print(now_layer_save_for_base)
        # 还是先写递推的写法,递归的写法想不清楚.
        # 建立layer信息
        layer1 = {}
        for i in range(len(data)):
            for jj in range(len(now_layer_save_for_data)):
                j = now_layer_save_for_data[jj]
                j2 = now_layer_save_for_base[jj]  # 光用汉字来做key会发生无法区分清华,中华这种bug.
                if data[i][0] == j:
                    layer1.setdefault((j, j2), [])
                    layer1[(j, j2)].append(i)
        # 用layer1,data里面的信息,对base里面信息进行加工,也就是如果单字就取反
        for i in layer1:
            if i[0] in data:
                base[i[1]] = -base[i[1]]

        # 搭建第二层:先找到将要被搭建的字
        # 利用last_layer和now_layer_save_for_data和now_layer_save_for_base来找.
        now_layer = last_layer + 1

        # for i in range(len(now_layer_save_for_data)):
        #    tmp=now_layer_save_for_data[i]#tmp就是清
        #    id=now_layer_save_for_base[i]#id 就是清的base数组里面的值
        # 找到清后面的字,也就是data里面第一个字为清的字.如果每建立一个节点就遍历一遍会是至少O(N方),并且
        # 基本严格大于这个数字,太大了.我想法是一层的东西同时处理,这样一层只遍历一次.降到线性搜索.
        # 对于同时一堆if,显然效率不行,所以还是字典来替代多if并列.还是慢,想到用类似线段树的手段来记录.
        # 里面的每一层用一个字典来表示,一个value是一个list
        # 根据layer1建立layer2
        layer = layer1
        print(layer)
        # 下面就可以建立layer2了#从这里就能分析出为什么要把上一层有同一个前缀的都建立完再弄下一个.
        # 下面整合起来是从一个layer得到这个层的全base数组和check数组.可以封装起来for循环.
        for iii in range(1, max_ceng):
            now_layer = iii + 1
            layer4 = {}
            print(layer)  # layer1:{('清', 2): [0, 1, 2], ('中', 7): [3], ('华', 3): [4]}

            for i in layer:
                lastword = i[0]
                lastindex = i[1]
                beixuan = layer[i]
                # 找到应该插入哪个
                charu = []
                # 把beixuan里面长度不够的剔除,他长度不够其实就表示已经在上一步是词组了.
                beixuan2 = []
                for i in beixuan:
                    if len(data[i]) >= now_layer:
                        beixuan2.append(i)
                beixuan = beixuan2

                for i in beixuan:
                    newword = data[i][now_layer - 1]
                    if newword not in charu:
                        charu.append(newword)
                # 把charu里面的东西进入base,check算法中

                now_layer_save_for_data = []  # 为了下一层的递推而记录的文字信息
                now_layer_save_for_base = []  # 为了下一层的递推而记录的index信息
                col_now = charu  # 插入华,新
                before_index = abs(lastindex)
                for i in col_now:

                    while 1:
                        index = abs(base[before_index]) + bianma[i]
                        if base[index] == '#':  # 说明没有人占用

                            break
                        else:
                            if base[before_index] > 0:
                                base[before_index] += 1
                            else:
                                base[before_index] -= 1
                            print(base)
                # 对于已经构成词汇的词语base里面的数要取相反数.
                beixuanciku = [data[i][now_layer - 1:] for i in beixuan]
                # 调试状态vs2017把鼠标放变量上就能看到他的取值,很放方便.任意位置都能看
                for i in col_now:
                    if i in beixuanciku:
                        index = abs(base[before_index]) + bianma[i]
                        base[index] = -abs(base[before_index])  # 注意这地方不能写-要写-abs
                        check[index] = before_index
                        now_layer_save_for_data.append(i)
                        now_layer_save_for_base.append(index)
                    else:
                        index = abs(base[before_index]) + bianma[i]
                        base[index] = base[before_index]
                        check[index] = before_index
                        now_layer_save_for_data.append(i)
                        now_layer_save_for_base.append(index)

                # 更新layer

                for i in beixuan:
                    for jj in range(len(now_layer_save_for_data)):
                        j = now_layer_save_for_data[jj]
                        j2 = now_layer_save_for_base[jj]  # 光用汉字来做key会发生无法区分清华,中华这种bug.
                        if data[i][now_layer - 1] == j:
                            layer4.setdefault((j, j2), [])
                            layer4[(j, j2)].append(i)

            # 已经得到了新的layer4,替换回去,为了递推.
            layer = layer4

        # 打印上个layer
        print(layer)  # {('清', 2): [0, 1, 2], ('中', 7): [3], ('华', 3): [4]} 上个layeer信息
        # 下面需要更新layer
        layernew = {}
        for i in layer:  # 逐个计算里面的对儿即可.比如先计算('清', 2): [0, 1, 2]应该改成什么
            pass

            # for jj in range(len(now_layer_save_for_data)):
            #  j=now_layer_save_for_data[jj]
            #  j2=now_layer_save_for_base[jj]#光用汉字来做key会发生无法区分清华,中华这种bug.
            #  if data[i][0]==j:
            #      layer1.setdefault((j,j2),[])
            #      layer1[(j,j2)].append(i)

        print(now_layer_save_for_data)
        print(now_layer_save_for_base)

        print('测试')  # 第二列也zhengque
        # 经过我2天超过20个小时的学习和写代码,写出了这个例子的base数组和check数组.修改些小bug就可以了.
        # 绝逼不比红黑树简单.网上也几乎没有代码实现.因为我主题layer是从第一层建立后针对2到n层开始建立的
        # 所以第一层如果是单字,直接返回这种情况,我还没写,但是相对盖起来简单.
        print(base)
        print(check)
        # 最后的最后,用self把结果传出去
        self.base = base
        self.check = check

    def search(self, a):  # 通过这个函数a在data是否存在,这个函数随便玩了

        tmp = 0
        # self写起来太麻烦,
        bianma = self.bianma
        base = self.base
        check = self.check
        i = a[0]
        if len(a) == 1:
            tmp = 1 + bianma[i]
            return base[tmp] < 0
        else:
            first = 1 + bianma[a[0]]
            for i in range(len(a) - 1):
                tmp = abs(base[first]) + bianma[a[i + 1]]
                if check[tmp] != first:
                    return False
                first = tmp
            return base[tmp] < 0


'''
base:[1, '#', -3, 2, -2, -3, -6, 2, -3, -2, -6, '#', '#']
check:['#', '#', 0, 0, 7, 2, 5, 0, 2, 3, 6, '#', '#']
'''

# 测试:
a = DAT(['清华', '清华大学', '清新', '中华', '华人', '清'])
# 进行search测试
print(a.search('清华大学'))
# 经过测试,稍微大一点的数据也是能跑出来的.