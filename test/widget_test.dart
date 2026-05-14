import 'package:flutter_test/flutter_test.dart';
import 'package:vortex_login_app/main.dart' as app;

void main() {
  testWidgets('app starts', (WidgetTester tester) async {
    app.main();
    await tester.pump();
    expect(tester.takeException(), isNull);
  });
}
