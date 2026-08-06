#include "ParamsIO.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
	bool IsFiniteFloat(float value)
	{
		return std::isfinite((double)value) != 0;
	}

	float ClampFloat(float value, float minValue, float maxValue)
	{
		return std::max(minValue, std::min(value, maxValue));
	}

	std::string Trim(const std::string& value)
	{
		const size_t first =
			value.find_first_not_of(" \t\r\n");
		if (first == std::string::npos)
		{
			return std::string();
		}
		const size_t last =
			value.find_last_not_of(" \t\r\n");
		return value.substr(first, last - first + 1);
	}

	// 解析 `key=value` 行；空行 / 注释行返回 false
	bool ReadKeyValue(const std::string& rawLine, std::string& key, std::string& value)
	{
		const std::string line = Trim(rawLine);
		if (line.empty() || line[0] == '#')
		{
			return false;
		}
		const size_t separator = line.find('=');
		if (separator == std::string::npos)
		{
			return false;
		}
		key = Trim(line.substr(0, separator));
		value = Trim(line.substr(separator + 1));
		return !key.empty();
	}

	bool ParseFloatValue(const std::string& text, float& result)
	{
		std::istringstream stream(text);
		stream >> result;
		return !stream.fail() && IsFiniteFloat(result);
	}

	bool ParseIntValue(const std::string& text, int& result)
	{
		std::istringstream stream(text);
		stream >> result;
		return !stream.fail();
	}

	// 解析 `bodyN{X,Y,Radius}` 键。component：0 = X，1 = Y，2 = Radius
	bool ParseBodyKey(const std::string& key, int& index, int& component)
	{
		if (key.compare(0, 4, "body") != 0)
		{
			return false;
		}
		size_t pos = 4;
		while (pos < key.size() && std::isdigit((unsigned char)key[pos]))
		{
			pos++;
		}
		if (pos == 4)
		{
			return false;
		}
		std::istringstream stream(key.substr(4, pos - 4));
		stream >> index;
		if (stream.fail() || index < 0 || index >= kMaxBodies)
		{
			return false;
		}
		const std::string suffix = key.substr(pos);
		if (suffix == "X") component = 0;
		else if (suffix == "Y") component = 1;
		else if (suffix == "Radius") component = 2;
		else return false;
		return true;
	}

	// §4.3 二次校验：坐标 clamp / 半径上限 / 两两不重叠 / bodyCount 截断。
	// 重叠或网格放不下的物体直接丢弃（数组前移），非法半径的物体回退丢弃。
	// 重叠判定按全局形状（请求 1）：圆=欧氏距离，方形=两轴均重叠，菱形=L1 距离。
	bool BodiesOverlapShape(
		float ax, float ay, float ar,
		float bx, float by, float br,
		ObstacleShape shape)
	{
		const float dx = std::fabs(ax - bx);
		const float dy = std::fabs(ay - by);
		switch (shape)
		{
		case ObstacleShape::Box:
			return dx < ar + br && dy < ar + br;
		case ObstacleShape::Diamond:
			return dx + dy < ar + br;
		case ObstacleShape::Circle:
		default:
			return dx * dx + dy * dy < (ar + br) * (ar + br);
		}
	}

	void SanitizeBodies(RigidBody* bodies, int& bodyCount,
		int nx, int ny, ObstacleShape shape)
	{
		bodyCount = std::max(0, std::min(bodyCount, kMaxBodies));
		const float maxRadius = std::min(
			20.0f,
			0.5f * ((float)std::min(nx, ny) - 2.0f));

		int kept = 0;
		for (int i = 0; i < bodyCount; i++)
		{
			RigidBody& body = bodies[i];
			if (!(body.radius > 0.0f) ||
				!IsFiniteFloat(body.radius) ||
				!IsFiniteFloat(body.x) ||
				!IsFiniteFloat(body.y))
			{
				continue;   // 非法 → 丢弃
			}

			body.radius = std::min(body.radius, maxRadius);
			const float margin = body.radius + 2.0f;
			const float maxX = (float)nx - margin - 1.0f;
			const float maxY = (float)ny - margin - 1.0f;
			if (maxX < margin || maxY < margin)
			{
				continue;   // 网格放不下 → 丢弃
			}
			body.x = ClampFloat(body.x, margin, maxX);
			body.y = ClampFloat(body.y, margin, maxY);
			body.tx = body.x;
			body.ty = body.y;
			body.vx = 0.0f;
			body.vy = 0.0f;

			bool overlaps = false;
			for (int j = 0; j < kept; j++)
			{
				const RigidBody& other = bodies[j];
				// 请求 1：按全局形状判定（+1 安全间距）
				if (BodiesOverlapShape(
					body.x, body.y, body.radius + 1.0f,
					other.x, other.y, other.radius + 1.0f,
					shape))
				{
					overlaps = true;
					break;
				}
			}
			if (overlaps)
			{
				continue;   // 与已保留物体重叠 → 丢弃
			}

			body.selected = (kept == 0);
			bodies[kept++] = body;
		}
		bodyCount = kept;
	}
}

bool SaveAppParams(
	const char* path,
	const DemoCaseDefinition& def,
	DemoFieldView view,
	int stepsPerFrame,
	bool smokeEnabled,
	ObstacleShape shape,
	int bodyCount,
	const RigidBody* bodies)
{
	std::ofstream out(path, std::ios::out | std::ios::trunc);
	if (!out.is_open())
	{
		return false;
	}

	out << "# Home2D LBM params (key=value per line, '#' = comment)\n";
	out << "case=" << (def.cliName != nullptr ? def.cliName : "karman") << "\n";
	out << "fieldView=" << (view == DemoFieldView::VelocityMagnitude ? 0 : 1) << "\n";
	out << "stepsPerFrame=" << stepsPerFrame << "\n";
	out << "smokeEnabled=" << (smokeEnabled ? 1 : 0) << "\n";
	out << "obstacleShape=" << (int)shape << "\n";
	out << "bodyCount=" << bodyCount << "\n";
	bodyCount = std::max(0, std::min(bodyCount, kMaxBodies));
	for (int i = 0; i < bodyCount; i++)
	{
		out << "body" << i << "X=" << bodies[i].x << "\n";
		out << "body" << i << "Y=" << bodies[i].y << "\n";
		out << "body" << i << "Radius=" << bodies[i].radius << "\n";
	}
	out << "viscosity=" << def.viscosity << "\n";
	out << "initialUx=" << def.initialUx << "\n";
	out << "initialUy=" << def.initialUy << "\n";
	out << "inletUx=" << def.inletUx << "\n";
	out << "inletUy=" << def.inletUy << "\n";
	out << "inletPerturbationAmplitude=" << def.inletPerturbationAmplitude << "\n";
	out << "inletPerturbationPeriod=" << def.inletPerturbationPeriod << "\n";
	out << "speedColorMax=" << def.speedColorMax << "\n";
	out << "vorticityColorMax=" << def.vorticityColorMax << "\n";
	out << "jetWidth=" << def.jetWidth << "\n";
	out.close();
	return true;
}

bool LoadAppParams(
	const char* path,
	DemoCaseId& caseId,
	DemoCaseDefinition& def,
	DemoFieldView& view,
	int& stepsPerFrame,
	bool& smokeEnabled,
	ObstacleShape& shape,
	int& bodyCount,
	RigidBody* bodies,
	bool applyFileCase)
{
	std::ifstream in(path);
	if (!in.is_open())
	{
		return false;
	}

	// 先缓存全部键值：case 键可能出现在任意位置，需先确定 case 再定基底 def
	typedef std::pair<std::string, std::string> KeyValue;
	std::vector<KeyValue> entries;
	std::string line;
	while (std::getline(in, line))
	{
		std::string key, value;
		if (ReadKeyValue(line, key, value))
		{
			entries.push_back(std::make_pair(key, value));
		}
	}
	in.close();
	if (entries.empty())
	{
		return true;   // 空文件视为合法（全部字段取默认）
	}

	// 1. 确定最终 case（§4.3 优先级：applyFileCase=false 时忽略 case 键）
	if (applyFileCase)
	{
		for (size_t i = 0; i < entries.size(); i++)
		{
			if (entries[i].first == "case")
			{
				DemoCaseId parsedId;
				if (ParseDemoCaseName(entries[i].second, parsedId))
				{
					caseId = parsedId;
				}
				break;
			}
		}
	}

	// 2. 以该 case 的默认 def 为基底（保证任何缺失字段都有合法值）
	def = GetDefaultDefinition(caseId);
	view = def.defaultView;
	stepsPerFrame = def.initialStepsPerFrame;
	smokeEnabled = false;
	shape = ObstacleShape::Circle;
	bodyCount = 0;
	for (int i = 0; i < kMaxBodies; i++)
	{
		bodies[i] = RigidBody();
	}

	// 3. 逐项覆盖；非法数值回退默认（§4.3）
	float bodyValues[kMaxBodies][3] = {};   // [i][0..2] = X / Y / Radius
	bool bodyPresent[kMaxBodies] = {};
	for (size_t i = 0; i < entries.size(); i++)
	{
		const std::string& key = entries[i].first;
		const std::string& value = entries[i].second;

		if (key == "case")
		{
			continue;   // 已处理
		}
		if (key == "fieldView")
		{
			int parsed = 0;
			if (ParseIntValue(value, parsed))
			{
				view = (parsed == 0)
					? DemoFieldView::VelocityMagnitude
					: DemoFieldView::Vorticity;
			}
		}
		else if (key == "stepsPerFrame")
		{
			int parsed = 0;
			if (ParseIntValue(value, parsed) && parsed >= 1 && parsed <= 50)
			{
				stepsPerFrame = parsed;
			}
		}
		else if (key == "smokeEnabled")
		{
			int parsed = 0;
			if (ParseIntValue(value, parsed))
			{
				smokeEnabled = (parsed != 0);
			}
		}
		else if (key == "obstacleShape")
		{
			int parsed = 0;
			if (ParseIntValue(value, parsed) && parsed >= 0 && parsed <= 2)
			{
				shape = (ObstacleShape)parsed;
			}
		}
		else if (key == "bodyCount")
		{
			int parsed = 0;
			if (ParseIntValue(value, parsed))
			{
				bodyCount = std::max(0, std::min(parsed, kMaxBodies));
			}
		}
		else if (key == "viscosity")
		{
			float parsed = 0.0f;
			if (ParseFloatValue(value, parsed) && parsed > 0.0f && parsed <= 0.05f)
			{
				def.viscosity = parsed;
			}
		}
		else if (key == "initialUx")
		{
			float parsed = 0.0f;
			if (ParseFloatValue(value, parsed))
			{
				def.initialUx = ClampFloat(parsed, -1.0f, 1.0f);
			}
		}
		else if (key == "initialUy")
		{
			float parsed = 0.0f;
			if (ParseFloatValue(value, parsed))
			{
				def.initialUy = ClampFloat(parsed, -1.0f, 1.0f);
			}
		}
		else if (key == "inletUx")
		{
			float parsed = 0.0f;
			if (ParseFloatValue(value, parsed))
			{
				// 请求 1：inlet ux 允许为负（负值 = 流向左偏）
				def.inletUx = ClampFloat(parsed, -0.2f, 0.2f);
			}
		}
		else if (key == "inletUy")
		{
			float parsed = 0.0f;
			if (ParseFloatValue(value, parsed))
			{
				def.inletUy = ClampFloat(parsed, 0.0f, 0.2f);
			}
		}
		else if (key == "inletPerturbationAmplitude")
		{
			float parsed = 0.0f;
			if (ParseFloatValue(value, parsed))
			{
				def.inletPerturbationAmplitude =
					ClampFloat(parsed, 0.0f, 0.01f);
			}
		}
		else if (key == "inletPerturbationPeriod")
		{
			int parsed = 0;
			if (ParseIntValue(value, parsed) && parsed >= 1 && parsed <= 10000)
			{
				def.inletPerturbationPeriod = parsed;
			}
		}
		else if (key == "speedColorMax")
		{
			float parsed = 0.0f;
			if (ParseFloatValue(value, parsed))
			{
				def.speedColorMax = ClampFloat(parsed, 0.01f, 0.5f);
			}
		}
		else if (key == "vorticityColorMax")
		{
			float parsed = 0.0f;
			if (ParseFloatValue(value, parsed))
			{
				def.vorticityColorMax = ClampFloat(parsed, 0.01f, 0.5f);
			}
		}
		else if (key == "jetWidth")
		{
			int parsed = 0;
			if (ParseIntValue(value, parsed) && parsed >= 2)
			{
				def.jetWidth = std::min(parsed, std::max(2, def.nx / 2));
			}
		}
		else
		{
			int bodyIndex = -1, component = -1;
			if (ParseBodyKey(key, bodyIndex, component))
			{
				float parsed = 0.0f;
				if (ParseFloatValue(value, parsed))
				{
					bodyValues[bodyIndex][component] = parsed;
					bodyPresent[bodyIndex] = true;
				}
			}
			// 未知键：忽略（不报错）
		}
	}

	// 4. 组装 bodies（仅保留 X/Y/Radius 三项齐全的物体），再二次校验
	int assembledCount = 0;
	for (int i = 0; i < kMaxBodies; i++)
	{
		if (!bodyPresent[i])
		{
			continue;
		}
		bodies[assembledCount] = RigidBody();
		bodies[assembledCount].x = bodyValues[i][0];
		bodies[assembledCount].y = bodyValues[i][1];
		bodies[assembledCount].radius = bodyValues[i][2];
		assembledCount++;
	}
	bodyCount = std::min(bodyCount, assembledCount);
	SanitizeBodies(bodies, bodyCount, def.nx, def.ny, shape);
	return true;
}
