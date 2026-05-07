import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vortex_login_app/main.dart';

void main() {
  testWidgets('login validates and signs in', (WidgetTester tester) async {
    await tester.pumpWidget(const VortexLoginApp());

    await tester.tap(find.text('Sign in'));
    await tester.pump();
    expect(find.text('Enter your email'), findsOneWidget);

    await tester.enterText(find.byType(TextFormField).at(0), 'user@example.com');
    await tester.enterText(find.byType(TextFormField).at(1), 'secret1');
    await tester.tap(find.text('Sign in'));
    await tester.pump();

    expect(find.text('Welcome'), findsOneWidget);
    expect(find.text('Signed in as user@example.com'), findsOneWidget);
  });
}
