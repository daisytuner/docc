#include "sdfg/codegen/utils.h"

#include <gtest/gtest.h>

using namespace sdfg;

TEST(PrettyPrinterTest, Indentation) {
    codegen::PrettyPrinter printer;
    printer << "Hello" << std::endl;
    printer.setIndent(4);
    printer << "World" << std::endl;
    printer.setIndent(0);
    printer << "!" << std::endl;

    EXPECT_EQ(printer.str(), "Hello\n    World\n!\n");
}

TEST(PrettyPrinterTest, Frozen) {
    codegen::PrettyPrinter printer(0, true);
    EXPECT_THROW(printer << "Hello", std::runtime_error);
}

TEST(PrettyPrinterTest, ContentIgnoresWhitespace) {
    codegen::PrettyPrinter printer(4);
    auto mark = printer.content();
    printer << "" << " " << '\n' << std::endl;
    EXPECT_EQ(printer.content(), mark);

    printer << "x = 1" << std::endl;
    EXPECT_GT(printer.content(), mark);

    mark = printer.content();
    printer << 0;
    EXPECT_GT(printer.content(), mark);
}
