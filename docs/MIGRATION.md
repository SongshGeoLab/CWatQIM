# 迁移指引

`cwatqim` 是带 DOI 的公开包（见 `.zenodo.json`、`sync-public-repo.yml`），所以每一次
删名字或改签名都可能打断外部引用——而外部引用是引用不到本仓 issue 的。本文件记录
那些变更，以及旧代码该怎么改。

体例上的一条原则：**能转发的留垫片并发告警，不能转发的留一个解释去向的桩，两者都不
静默消失。** 已有先例是 `core/payoff.py` 为 `lost_reputation` 留的那个垫片。

## v0.2 — 社会项进入决策（issue #94、#110、#72、#130）

这一批变更来自把社会项从事后加权改成进入配水决策本身。

### 求解器改名，且换了算法

| 旧 | 新 |
| --- | --- |
| `cwatqim.core.allocation.optimize_surface_share` | `cwatqim.core.allocation.solve_surface_share` |

旧名字仍可调用，会发 `DeprecationWarning` 后转发。

**改名不只是改名**：目标函数对 `q_surface` 是分段仿射的（作物收入在求解前就绑死，
水费线性，社会项只在配额处跳一次），极大值必在端点，所以解法从差分进化换成了枚举
端点的 closed form。结果因此是**精确**的，且不再消耗模型的随机数流。

原先差分进化留下的收敛残差不小：非角点解到最近角点的相对距离中位数是 0.0009。

### 差分进化的调参入口全部失效

`DE_DEFAULTS`、`DE_RNG_KWARG` 已删除，按属性取它们会抛出说明去向的 `AttributeError`。
没有种群、没有代数、也没有随机数可播种，所以**没有等价物可转发**。

`solve_surface_share` 带 `**payoff_kwargs`，因此旧调用方传 `ga_kwargs=` / `rng=` /
`popsize=` 之类**不会报错**——它们会被原样转给目标函数。这是最危险的一格：调用方以为
在调参，实际什么都没发生。现在这些关键字会被摘掉并发 `DeprecationWarning`。

```python
# 旧（v0.1.6 及更早：City.water_withdraw 接受 ga_kwargs）
city.water_withdraw(ga_kwargs={"popsize": 30})

# 新——调参参数删掉即可，解是精确的
city.water_withdraw()
```

### 举报不再是抽签

| 删除 | 改用 |
| --- | --- |
| `City.hate_a_behave(behave)` | `City.will_report(behave, my_decision)` |

按旧名字取属性会抛出说明去向的 `AttributeError`。

⚠️ **新方法多一个必填参数**：`my_decision` 是本主体自己这一年的决定，没有默认值。
确定性规则下「谁在批评谁」取决于双方的决定，而旧的抽签版本只看被观察方，所以这个
参数不能省——省了会直接 `TypeError`。

规则从伯努利抽签变成了确定性阈值 `v > 1 − grid`，所以一条边的两端**必然**对同一次事件
给出相同判断。旧的 `_judgements` 表为同一条边的两端各存一份独立随机账本，两者可以互相
矛盾——那正是删掉它的原因（issue #72）。纯函数版本是
`cwatqim.core.payoff.reports_defector(vengefulness, grid)`。

### `City.agg_payoff` 新增必填参数 `revenue`

**故意不给默认值。** 效用本身是 `U = e · s`，`revenue` 不进它；但它是被采集的一列，
分析侧靠它把经济诱惑与威慑放到同一根轴上。默认成 `0.0` 会往那一列里写一个错的数，
而且没有任何提示。宁可 `TypeError` 立刻炸掉。

```python
# 旧
u = city.agg_payoff(e, s)

# 新——把毛收入一起传进去
u = city.agg_payoff(e, s, revenue=city.revenue)
```

### `City.water_withdraw` 的 `ga_kwargs` 形参删除

同上，该关键字现在会被摘掉并告警。

## v0.1.x — 社会得分的方向（issue #60）

| 旧 | 新 |
| --- | --- |
| `cwatqim.core.payoff.lost_reputation` | `cwatqim.core.payoff.social_standing` |

旧名字仍可调用并发告警。**注意方向**：返回的是**保住**的社会得分（1.0 = 无人批评，
0.0 = 尽失），不是要减去的损失。旧名字描述的是它的补集，据此写出的
`payoff = e * (1 - s)` 会把机制整个反过来。
