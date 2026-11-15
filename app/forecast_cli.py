"""命令行入口：冻结预测、结算或历史覆盖回放。"""
import json
import sys
from pathlib import Path

from .forecast import inspect_forecast, replay, settle
from .store import FrozenStore

USAGE = (
    '用法:\n'
    '  python -m app.forecast_cli forecast DATA.json [--config CONFIG.json]\n'
    '  python -m app.forecast_cli replay W1.json W2.json ... [--config CONFIG.json]\n'
    '  python -m app.forecast_cli settle FROZEN.json LATER.json'
)


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _emit(value):
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    try:
        sys.stdout.write(payload + '\n')
        sys.stdout.flush()
    except BrokenPipeError:
        # 下游提前关闭管道（例如 head）：丢弃剩余输出并用 devnull 顶替，
        # 避免解释器退出时的二次 BrokenPipeError。
        import os

        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ('forecast', 'replay', 'settle'):
        raise SystemExit(USAGE)
    command = argv.pop(0)
    config = None
    if '--config' in argv:
        position = argv.index('--config')
        if position + 1 >= len(argv):
            raise SystemExit('--config需要文件路径')
        config = _load(argv[position + 1])
        del argv[position:position + 2]
    if not argv:
        raise SystemExit('请指定资料JSON路径')

    if command == 'forecast':
        if len(argv) != 1:
            raise SystemExit('forecast只接受一个资料路径，配置请用--config')
        _emit(
            inspect_forecast(_load(argv[0]), config=config, store=FrozenStore())
        )
        return

    if command == 'replay':
        if len(argv) < 2:
            raise SystemExit('replay至少需要两个时点快照')
        _emit(replay([_load(path) for path in argv], config=config))
        return

    if len(argv) != 2:
        raise SystemExit('settle需要预测JSON与结算资料JSON')
    if config is not None:
        raise SystemExit('settle不接受--config')
    _emit(settle(_load(argv[0]), _load(argv[1])))


if __name__ == '__main__':
    main()
