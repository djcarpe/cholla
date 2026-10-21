defmodule Cholla.GroundingTest do
  use ExUnit.Case, async: true

  test "numbers written in the sources ground argument paths" do
    args = %{
      "total" => 1250.0,
      "qty" => 3,
      "items" => [%{"price" => 9.99}],
      "year" => 2026,
      "ok" => true
    }

    paths =
      Cholla.grounded_number_paths(args, [
        "Pay $1,250.00 for 3 items at 9.99",
        "date: 2026-10-20 Tue 09:05"
      ])

    assert Enum.sort(paths) == ["items[0].price", "qty", "total"]
  end
end
