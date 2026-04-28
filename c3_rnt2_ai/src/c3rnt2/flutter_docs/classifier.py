from __future__ import annotations

import re
from dataclasses import dataclass


TAXONOMY = [
    "flutter_basics",
    "dart_for_flutter",
    "widgets",
    "layout",
    "constraints",
    "rendering",
    "state_management",
    "navigation",
    "forms_validation",
    "async_futures_streams",
    "networking",
    "persistence",
    "assets_images",
    "animations",
    "gestures",
    "accessibility",
    "internationalization",
    "theming",
    "adaptive_responsive",
    "platform_integration",
    "plugins",
    "testing_unit_widget_integration",
    "golden_tests",
    "performance",
    "devtools",
    "debugging",
    "build_release",
    "android_ios_deploy",
    "web_desktop",
    "architecture",
    "clean_architecture",
    "error_handling",
    "security",
    "packages",
    "api_reference",
    "migration_breaking_changes",
]


SPECIAL_TOPICS = {
    "RenderBox was not laid out": "constraints",
    "RenderFlex overflow": "layout",
    "RenderFlex overflowed": "layout",
    "constraints": "constraints",
    "responsive": "adaptive_responsive",
    "adaptive": "adaptive_responsive",
    "jank": "performance",
    "reusable widget": "widgets",
    "maintainable": "architecture",
    "widget test": "testing_unit_widget_integration",
    "golden test": "golden_tests",
    "DevTools": "devtools",
}


@dataclass(frozen=True)
class TopicRule:
    topic: str
    pattern: re.Pattern[str]
    difficulty: str = "intermediate"


RULES = [
    TopicRule("constraints", re.compile(r"(?i)constraint|bounded|unbounded|boxconstraint|renderbox")),
    TopicRule("layout", re.compile(r"(?i)layout|row|column|flex|expanded|flexible|overflow|listview|gridview|sliver")),
    TopicRule("rendering", re.compile(r"(?i)renderobject|rendering|paint|pipeline|compositor|layer tree")),
    TopicRule("state_management", re.compile(r"(?i)state management|provider|riverpod|bloc|setstate|inheritedwidget|change.?notifier")),
    TopicRule("navigation", re.compile(r"(?i)navigation|navigator|router|route|gorouter|deep link")),
    TopicRule("forms_validation", re.compile(r"(?i)form|textfield|text field|validation|validator|input")),
    TopicRule("async_futures_streams", re.compile(r"(?i)future|stream|async|await|isolate")),
    TopicRule("networking", re.compile(r"(?i)http|network|fetch|rest|websocket|socket")),
    TopicRule("persistence", re.compile(r"(?i)persist|storage|sqlite|shared_preferences|file")),
    TopicRule("assets_images", re.compile(r"(?i)asset|image|picture|font|icon")),
    TopicRule("animations", re.compile(r"(?i)animation|animated|tween|curve|ticker")),
    TopicRule("gestures", re.compile(r"(?i)gesture|tap|drag|pointer|mouse|touch")),
    TopicRule("accessibility", re.compile(r"(?i)accessibility|semantics|screen reader|a11y")),
    TopicRule("internationalization", re.compile(r"(?i)localization|internationalization|i18n|locale|intl")),
    TopicRule("theming", re.compile(r"(?i)theme|color scheme|material 3|typography")),
    TopicRule("adaptive_responsive", re.compile(r"(?i)responsive|adaptive|mediaquery|layoutbuilder|breakpoint|desktop|tablet")),
    TopicRule("platform_integration", re.compile(r"(?i)platform channel|methodchannel|android|ios|embedder|native")),
    TopicRule("plugins", re.compile(r"(?i)plugin|package|pubspec|pub.dev")),
    TopicRule("testing_unit_widget_integration", re.compile(r"(?i)test|testing|widgettest|integration_test|flutter_test|mock")),
    TopicRule("golden_tests", re.compile(r"(?i)golden")),
    TopicRule("performance", re.compile(r"(?i)performance|jank|profile|rebuild|frame|memory|raster")),
    TopicRule("devtools", re.compile(r"(?i)devtools|inspector|timeline|profiler")),
    TopicRule("debugging", re.compile(r"(?i)debug|exception|error|diagnos|troubleshoot|stack trace")),
    TopicRule("build_release", re.compile(r"(?i)build|release|apk|appbundle|ipa|obfuscate")),
    TopicRule("android_ios_deploy", re.compile(r"(?i)android|ios|play store|app store|xcode|gradle")),
    TopicRule("web_desktop", re.compile(r"(?i)web|desktop|windows|macos|linux")),
    TopicRule("architecture", re.compile(r"(?i)architecture|repository|datasource|data layer|domain|presentation|feature")),
    TopicRule("clean_architecture", re.compile(r"(?i)clean architecture|entity|use case|usecase|dto")),
    TopicRule("error_handling", re.compile(r"(?i)error handling|exception|try|catch|failure")),
    TopicRule("security", re.compile(r"(?i)security|auth|token|permission|secure|privacy")),
    TopicRule("packages", re.compile(r"(?i)package|dependency|pubspec|pub add")),
    TopicRule("api_reference", re.compile(r"(?i)api reference|class |method |property |constructor |library")),
    TopicRule("migration_breaking_changes", re.compile(r"(?i)migration|breaking change|deprecated|upgrade")),
    TopicRule("dart_for_flutter", re.compile(r"(?i)\bdart\b|language|null safety|extension|mixin|class")),
    TopicRule("widgets", re.compile(r"(?i)widget|statelesswidget|statefulwidget|build method|materialapp|scaffold")),
]


def classify_text(text: str, *, url: str = "", title: str = "") -> tuple[str, str]:
    haystack = f"{url}\n{title}\n{text}"
    path_topic = _topic_from_url(url)
    if path_topic:
        return path_topic, _difficulty(haystack)
    for literal, topic in SPECIAL_TOPICS.items():
        if literal.lower() in haystack.lower():
            return topic, _difficulty(haystack)
    for rule in RULES:
        if rule.pattern.search(haystack):
            return rule.topic, _difficulty(haystack)
    return "flutter_basics", _difficulty(haystack)


def _topic_from_url(url: str) -> str | None:
    lower = url.lower()
    mapping = [
        ("app-architecture", "architecture"),
        ("clean-architecture", "clean_architecture"),
        ("ui/layout/constraints", "constraints"),
        ("ui/layout", "layout"),
        ("ui/widgets/assets", "assets_images"),
        ("ui/widgets/animation", "animations"),
        ("ui/widgets", "widgets"),
        ("ui/adaptive-responsive", "adaptive_responsive"),
        ("ui/accessibility", "accessibility"),
        ("ui/assets", "assets_images"),
        ("ui/animations", "animations"),
        ("ui/interactivity/gestures", "gestures"),
        ("cookbook/forms", "forms_validation"),
        ("cookbook/networking", "networking"),
        ("cookbook/navigation", "navigation"),
        ("cookbook/persistence", "persistence"),
        ("cookbook/testing/golden", "golden_tests"),
        ("cookbook/testing", "testing_unit_widget_integration"),
        ("testing/code-debugging", "debugging"),
        ("testing/native-debugging", "debugging"),
        ("testing/", "testing_unit_widget_integration"),
        ("data-and-backend/state-mgmt", "state_management"),
        ("performance", "performance"),
        ("tools/devtools", "devtools"),
        ("internationalization", "internationalization"),
        ("ui/design/themes", "theming"),
        ("platform-integration", "platform_integration"),
        ("packages-and-plugins", "plugins"),
        ("deployment/android", "android_ios_deploy"),
        ("deployment/ios", "android_ios_deploy"),
        ("deployment", "build_release"),
        ("platforms/web", "web_desktop"),
        ("platforms/desktop", "web_desktop"),
        ("api.flutter.dev", "api_reference"),
        ("migration", "migration_breaking_changes"),
        ("dart.dev", "dart_for_flutter"),
    ]
    for needle, topic in mapping:
        if needle in lower:
            return topic
    return None


def classify_chunk(chunk: dict) -> dict:
    topic, difficulty = classify_text(
        str(chunk.get("text") or ""),
        url=str(chunk.get("url") or ""),
        title=str(chunk.get("title") or ""),
    )
    out = dict(chunk)
    out["topic"] = topic
    out["difficulty"] = difficulty
    return out


def _difficulty(text: str) -> str:
    advanced = re.compile(r"(?i)renderobject|platform channel|performance|isolate|sliver|custom painter|ffi|embedder")
    beginner = re.compile(r"(?i)get started|install|hello world|basic|introduction|first app")
    if advanced.search(text):
        return "advanced"
    if beginner.search(text):
        return "beginner"
    return "intermediate"
